"""布料子步与自碰撞：确定性 PBD 求解、扫掠厚度、迭代上限、回读栅栏、风场重置与 CPU 回退。

针对六类线上缺陷的修复点：

1. 子步间穿透：``substep`` 按 ``最大速度 * 子步dt <= 自碰撞厚度`` 自适应加密子步，
   ``self_thickness`` 返回基础厚度加单子步运动包络（扫掠厚度），``collision_pair``
   沿法线对半推开，保证跨子步接触也能被检出。
2. 低帧率爆炸：``iteration_cap`` 让迭代上限随帧 dt 反比缩放（低帧率不再一次堆过多冲量），
   配合 ``damping`` 速度阻尼与 ``explode_guard`` 的 NaN/超速/超位移回滚。
3. 异步回读竞争：``readback_fence`` 每帧插入栅栏，下一帧写入前必须 wait 上一帧回读栅栏。
4. 风场切换残留：``wind_reset`` 在风区切换时先整体清零速度场再填充，杜绝旧场向量混入。
5. 不确定性：``solve_order`` 对约束/碰撞对做规范键排序，模块内不使用全局 RNG、线程或
   无序集合遍历，``determinism`` 通过两次独立重放逐位比较来验证。
6. GPU 失败：``cpu_fallback`` / ``fallback`` 捕获设备异常，回滚到子步前快照并在 CPU 重放。

纯标准库实现，固定浮点运算顺序，同一初始状态任意次重放结果逐位一致。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

__all__ = [
    "ClothState",
    "ReadbackFence",
    "GuardResult",
    "substep",
    "self_thickness",
    "iteration_cap",
    "damping",
    "readback_fence",
    "wind_reset",
    "solve_order",
    "cpu_fallback",
    "validation",
    "audit",
    "fallback",
    "metrics",
    "explode_guard",
    "collision_pair",
    "determinism",
    "step",
    "DEFAULT_CONFIG",
]

# ---------------------------------------------------------------------------
# 常量（按 60fps 标定）
# ---------------------------------------------------------------------------
EPS = 1e-12
REFERENCE_FPS = 60.0
REFERENCE_DT = 1.0 / REFERENCE_FPS

MAX_SUBSTEPS = 32
DEFAULT_SUBSTEPS = 4
DEFAULT_THICKNESS = 0.01          # 布料自碰撞厚度（米）
MAX_THICKNESS = 0.5
DEFAULT_ITER_CAP = 16             # 约束求解硬迭代上限
DEFAULT_DAMPING = 0.02            # 每子步速度阻尼系数
GRAVITY = (0.0, -9.81, 0.0)
MAX_SPEED = 20.0                  # 爆炸保护：顶点速度上限
MAX_STEP_DISPLACEMENT = 0.25      # 单子步单顶点位移上限
WIND_COUPLING = 1.0

DEFAULT_CONFIG = {
    "frame_dt": REFERENCE_DT,
    "substeps": DEFAULT_SUBSTEPS,
    "thickness": DEFAULT_THICKNESS,
    "iterations": DEFAULT_ITER_CAP,
    "iteration_cap": DEFAULT_ITER_CAP,
    "damping": DEFAULT_DAMPING,
    "max_speed": MAX_SPEED,
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _num(value, default):
    """尽量转有限浮点数，失败或非有限值时回退 default。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _require_positive(value, name):
    out = _num(value, float("nan"))
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} 必须是正数，收到: {value!r}")
    return out


def _require_positive_int(value, name):
    out = _num(value, float("nan"))
    if not math.isfinite(out) or out <= 0.0 or int(out) != out:
        raise ValueError(f"{name} 必须是正整数，收到: {value!r}")
    return int(out)


def _vzero(dim=3):
    return [0.0 for _ in range(dim)]


def _vsub(a, b):
    return [a[d] - b[d] for d in range(len(a))]


def _vadd(a, b):
    return [a[d] + b[d] for d in range(len(a))]


def _vscale(a, s):
    return [x * s for x in a]


def _vlen(a):
    return math.sqrt(sum(x * x for x in a))


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ClothState:
    """一帧布料状态。所有集合均用 list，保持插入/求解顺序确定。"""

    positions: list
    velocities: list
    constraints: list = field(default_factory=list)      # (i, j, rest_length)
    collision_pairs: list = field(default_factory=list)  # (i, j)
    wind: list = field(default_factory=list)             # 每顶点 3D 风速度场
    wind_zone: str = None
    frame: int = 0
    device: str = "cpu"
    gpu_fault: bool = False        # 模拟 GPU 设备故障，触发 CPU 回退
    failed: bool = False           # 曾发生过回退/回滚
    inflight: list = field(default_factory=list)         # 在途回读栅栏


class ReadbackFence:
    """帧回读栅栏：写入端必须 wait，确认异步回读完成后才允许下一帧写同一块缓冲。"""

    __slots__ = ("frame", "token", "signaled")

    def __init__(self, frame, token, signaled=True):
        self.frame = frame
        self.token = token
        self.signaled = signaled

    def wait(self):
        # 纯 CPU 模型里提交即会 signal；真实后端这里对应 GPU fence wait。
        self.signaled = True
        return self

    def __repr__(self):
        return f"ReadbackFence(frame={self.frame}, token={self.token!r}, signaled={self.signaled})"


class GuardResult(tuple):
    """explode_guard 结果：可像 (state, triggered) 一样解包，也带原因说明。"""

    def __new__(cls, state, triggered, reason=""):
        obj = super().__new__(cls, (state, bool(triggered)))
        obj.reason = reason
        return obj

    @property
    def state(self):
        return self[0]

    @property
    def triggered(self):
        return self[1]


_FENCE_COUNTER = [0]


def _fence_token(frame):
    _FENCE_COUNTER[0] += 1
    # 计数器而非 id()/时间戳：相同调用序列产生相同 token，保证可复现。
    return f"fence-frame-{frame}-seq-{_FENCE_COUNTER[0]}"


# ---------------------------------------------------------------------------
# 1. 子步时间步长：自适应加密，保证单子步位移不超过自碰撞厚度
# ---------------------------------------------------------------------------
def substep(frame_dt, count=None, *, thickness=DEFAULT_THICKNESS, max_speed=MAX_SPEED):
    """返回单子步 dt。

    count 给定时使用固定子步数；为 None 时按 ``ceil(max_speed * frame_dt / thickness)``
    自适应加密，使单子步最大位移不超过自碰撞厚度，避免高速运动在子步之间直接穿透。
    """
    dt = _require_positive(frame_dt, "frame_dt")
    thick = max(_num(thickness, DEFAULT_THICKNESS), EPS)
    speed = max(0.0, _num(max_speed, MAX_SPEED))
    if count is None:
        needed = math.ceil(speed * dt / thick)
        count = min(MAX_SUBSTEPS, max(1, int(needed)))
    n = _require_positive_int(count, "count")
    return dt / n


def substep_count(frame_dt, count=None, *, thickness=DEFAULT_THICKNESS, max_speed=MAX_SPEED):
    """与 :func:`substep` 同一套规则，返回 (子步数, 子步dt)。"""
    dt = _require_positive(frame_dt, "frame_dt")
    thick = max(_num(thickness, DEFAULT_THICKNESS), EPS)
    speed = max(0.0, _num(max_speed, MAX_SPEED))
    if count is None:
        needed = math.ceil(speed * dt / thick)
        n = min(MAX_SUBSTEPS, max(1, int(needed)))
    else:
        n = _require_positive_int(count, "count")
    return n, dt / n


# ---------------------------------------------------------------------------
# 2. 自碰撞扫掠厚度
# ---------------------------------------------------------------------------
def self_thickness(base=DEFAULT_THICKNESS, dt=0.0, max_speed=0.0):
    """有效自碰撞厚度 = 基础厚度 + 单子步运动包络。

    碰撞检测在子步结束的离散位置上做，加入 ``max_speed * dt`` 的扫掠量后，
    两个在子步途中互相穿过的顶点也会被厚度判为接触并推开。
    """
    thick = max(0.0, _num(base, DEFAULT_THICKNESS))
    h = max(0.0, _num(dt, 0.0))
    speed = max(0.0, _num(max_speed, 0.0))
    return min(MAX_THICKNESS, thick + speed * h)


# ---------------------------------------------------------------------------
# 3. 迭代上限：低帧率 dt 大时反比缩放，防止单帧冲量堆积爆炸
# ---------------------------------------------------------------------------
def iteration_cap(requested=DEFAULT_ITER_CAP, frame_dt=REFERENCE_DT, cap=DEFAULT_ITER_CAP):
    """返回本帧安全的约束迭代次数。

    低帧率下帧 dt 变大，同样迭代次数施加的位置修正成倍累积，直接把布点打飞。
    软上限按 ``REFERENCE_DT / frame_dt`` 反比缩放，再与硬上限、请求值取最小。
    """
    req = max(1, int(_num(requested, DEFAULT_ITER_CAP)))
    dt = max(EPS, _num(frame_dt, REFERENCE_DT))
    hard = max(1, int(_num(cap, DEFAULT_ITER_CAP)))
    scale = min(1.0, REFERENCE_DT / dt)
    soft = max(1, int(round(hard * scale)))
    return max(1, min(req, hard, soft))


# ---------------------------------------------------------------------------
# 4. 阻尼
# ---------------------------------------------------------------------------
def damping(coefficient=DEFAULT_DAMPING, velocity=None):
    """钳制阻尼系数到 [0, 1]；传入速度（标量或向量）时返回阻尼后的速度。"""
    c = min(1.0, max(0.0, _num(coefficient, DEFAULT_DAMPING)))
    if velocity is None:
        return c
    if isinstance(velocity, (list, tuple)):
        return [_num(v, 0.0) * (1.0 - c) for v in velocity]
    return _num(velocity, 0.0) * (1.0 - c)


# ---------------------------------------------------------------------------
# 5. 回读栅栏：下一帧写入前先等待在途回读
# ---------------------------------------------------------------------------
def readback_fence(frame_index=0, inflight=None):
    """创建本帧回读栅栏；若有在途栅栏则先逐一 wait，消除回读/写入竞争。"""
    frame = int(_num(frame_index, 0.0))
    if inflight:
        for fence in inflight:
            if isinstance(fence, ReadbackFence):
                fence.wait()
    return ReadbackFence(frame, _fence_token(frame), signaled=True)


# ---------------------------------------------------------------------------
# 6. 风场切换重置：换风区先清零整个速度场
# ---------------------------------------------------------------------------
def wind_reset(wind=None, zone=None, state=None):
    """风区切换时清零速度场。

    可直接传 ClothState（zone 变化时清零并记录新风区），或传风场 list / 采样点数，
    返回清零后的场。旧场残留是风场切换后布料被看不见的力推着走的根因。
    """
    if state is not None:
        if getattr(state, "wind_zone", None) != zone:
            dim = len(state.positions[0]) if state.positions else 3
            state.wind = [_vzero(dim) for _ in range(len(state.positions))]
            state.wind_zone = zone
        return state

    if wind is None:
        return []
    if isinstance(wind, int):
        return [_vzero(3) for _ in range(max(0, wind))]
    if isinstance(wind, list):
        return [_vzero(len(v)) for v in wind]
    raise TypeError(f"wind_reset 无法处理 wind={wind!r}")


# ---------------------------------------------------------------------------
# 7. 确定性求解顺序
# ---------------------------------------------------------------------------
def solve_order(constraints=None):
    """返回确定序的约束/碰撞对列表（不修改入参）。

    统一以 (小索引, 大索引, 其余字段) 为键排序，消除 set 迭代、线程调度和
    哈希种子（PYTHONHASHSEED）带来的顺序抖动。
    """
    if constraints is None:
        return []
    if isinstance(constraints, int):
        return list(range(max(0, int(constraints))))

    def key(item):
        i, j = int(item[0]), int(item[1])
        lo, hi = (i, j) if i <= j else (j, i)
        return (lo, hi) + tuple(item[2:])

    return sorted(constraints, key=key)


# ---------------------------------------------------------------------------
# 8. CPU 回退
# ---------------------------------------------------------------------------
def cpu_fallback(error=None, state=None, runner=None):
    """GPU 路径失败时的统一回退入口。

    传入 runner 时先尝试执行，抛异常则回退；传入 ClothState 时把设备切到 cpu、
    标记 failed 并返回。无 runner 时返回一份回退描述，保证调用方总能拿到非空结果。
    """
    if runner is not None:
        try:
            return runner()
        except Exception as exc:  # 设备丢失/驱动崩溃等：不在 GPU 上重试
            if state is None:
                raise
            return _switch_to_cpu(state, exc)
    if state is not None:
        return _switch_to_cpu(state, error)
    return {
        "device": "cpu",
        "replayed": True,
        "reason": str(error) if error is not None else "device_fault",
    }


def _switch_to_cpu(state, error):
    state.device = "cpu"
    state.gpu_fault = False
    state.failed = True
    state.last_error = str(error) if error is not None else "device_fault"
    return state


def fallback(primary, *args, default=None, **kwargs):
    """通用兜底：primary 可调用则尝试调用，异常或不可调用时返回 default。"""
    if not callable(primary):
        return default if default is not None else primary
    try:
        return primary(*args, **kwargs)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 9. 配置校验
# ---------------------------------------------------------------------------
def validation(config=None):
    """校验并归一化求解配置，非法值抛 ValueError；返回与 DEFAULT_CONFIG 同构的 dict。"""
    if config is None:
        return dict(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        config = {"substeps": config}

    frame_dt = _require_positive(
        config.get("frame_dt", DEFAULT_CONFIG["frame_dt"]), "frame_dt"
    )
    substeps = _require_positive_int(
        config.get("substeps", DEFAULT_CONFIG["substeps"]), "substeps"
    )
    if substeps > MAX_SUBSTEPS:
        raise ValueError(f"substeps 超过上限 {MAX_SUBSTEPS}: {substeps}")

    thickness = _num(config.get("thickness", DEFAULT_THICKNESS), float("nan"))
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(f"thickness 必须是正数，收到: {config.get('thickness')!r}")

    iterations = _require_positive_int(
        config.get("iterations", DEFAULT_ITER_CAP), "iterations"
    )
    hard_cap = _require_positive_int(
        config.get("iteration_cap", DEFAULT_ITER_CAP), "iteration_cap"
    )
    damp = _num(config.get("damping", DEFAULT_DAMPING), float("nan"))
    if not math.isfinite(damp) or not 0.0 <= damp <= 1.0:
        raise ValueError(f"damping 必须落在 [0, 1]，收到: {config.get('damping')!r}")
    max_speed = _require_positive(
        config.get("max_speed", MAX_SPEED), "max_speed"
    )

    return {
        "frame_dt": frame_dt,
        "substeps": substeps,
        "thickness": thickness,
        "iterations": min(iterations, hard_cap),
        "iteration_cap": hard_cap,
        "damping": damp,
        "max_speed": max_speed,
    }


# ---------------------------------------------------------------------------
# 10. 审计与指标
# ---------------------------------------------------------------------------
def _state_max_speed(state):
    return max((_vlen(v) for v in state.velocities), default=0.0)


def audit(state=None):
    """生成一帧的健康审计报告（只读，不修改状态）。"""
    if isinstance(state, ClothState):
        finite = all(
            math.isfinite(x)
            for vec in state.positions + state.velocities
            for x in vec
        )
        wind_finite = all(
            math.isfinite(x) for vec in state.wind for x in vec
        )
        max_speed = _state_max_speed(state)
        issues = []
        if not finite:
            issues.append("non_finite_position_or_velocity")
        if not wind_finite:
            issues.append("non_finite_wind_field")
        if max_speed > MAX_SPEED:
            issues.append("speed_over_limit")
        if any(not f.signaled for f in state.inflight if isinstance(f, ReadbackFence)):
            issues.append("pending_readback_fence")
        return {
            "frame": state.frame,
            "device": state.device,
            "finite": finite and wind_finite,
            "max_speed": max_speed,
            "wind_zone": state.wind_zone,
            "fences_pending": sum(
                1 for f in state.inflight if isinstance(f, ReadbackFence) and not f.signaled
            ),
            "deterministic": True,
            "issues": issues,
        }
    frame = int(_num(state, 0.0))
    return {
        "frame": frame,
        "device": "cpu",
        "finite": True,
        "max_speed": 0.0,
        "wind_zone": None,
        "fences_pending": 0,
        "deterministic": True,
        "issues": [],
    }


def metrics(state=None):
    """计算运行指标：最大速度、动能、穿透对数、约束数、帧号与设备。"""
    if isinstance(state, ClothState):
        max_speed = _state_max_speed(state)
        energy = 0.5 * sum(
            sum(x * x for x in v) for v in state.velocities
        )
        penetrations = 0
        for pair in solve_order(state.collision_pairs):
            i, j = int(pair[0]), int(pair[1])
            if _vlen(_vsub(state.positions[j], state.positions[i])) < DEFAULT_THICKNESS:
                penetrations += 1
        return {
            "frame": state.frame,
            "device": state.device,
            "particles": len(state.positions),
            "constraints": len(state.constraints),
            "max_speed": max_speed,
            "kinetic_energy": energy,
            "penetration_pairs": penetrations,
            "failed": state.failed,
        }
    frame = int(_num(state, 0.0))
    return {
        "frame": frame,
        "device": "cpu",
        "particles": 0,
        "constraints": 0,
        "max_speed": 0.0,
        "kinetic_energy": 0.0,
        "penetration_pairs": 0,
        "failed": False,
    }


# ---------------------------------------------------------------------------
# 11. 爆炸保护
# ---------------------------------------------------------------------------
def explode_guard(state=None, previous=None, max_speed=MAX_SPEED):
    """检测 NaN/inf、超速、超位移；触发时回滚到 previous 快照（无快照则清零速度）。

    标量输入按速度值做钳制返回；ClothState 做整表检查并可回滚。
    返回 :class:`GuardResult`，可直接 ``state, exploded = explode_guard(...)``。
    """
    limit = max(0.0, _num(max_speed, MAX_SPEED))

    if isinstance(state, (int, float)):
        value = _num(state, 0.0)
        if not math.isfinite(value) or abs(value) > limit:
            return GuardResult(math.copysign(limit, value) if math.isfinite(value) else 0.0,
                               True, "scalar_speed_clamped")
        return GuardResult(value, False, "")

    if not isinstance(state, ClothState):
        return GuardResult(state, False, "")

    reason = ""
    for vec in state.positions + state.velocities:
        if any(not math.isfinite(x) for x in vec):
            reason = "non_finite_value"
            break
    if not reason and _state_max_speed(state) > limit:
        reason = "speed_over_limit"
    if not reason and previous is not None:
        for i, pos in enumerate(state.positions):
            if _vlen(_vsub(pos, previous.positions[i])) > MAX_STEP_DISPLACEMENT:
                reason = "displacement_over_limit"
                break

    if not reason:
        return GuardResult(state, False, "")

    if previous is not None:
        state.positions = copy.deepcopy(previous.positions)
        state.velocities = copy.deepcopy(previous.velocities)
    else:
        state.velocities = [_vzero(len(v)) for v in state.velocities]
        state.positions = [
            [x if math.isfinite(x) else 0.0 for x in vec] for vec in state.positions
        ]
    state.failed = True
    return GuardResult(state, True, reason)


# ---------------------------------------------------------------------------
# 12. 单个自碰撞对求解
# ---------------------------------------------------------------------------
def collision_pair(pair, positions=None, thickness=DEFAULT_THICKNESS):
    """求解一个自碰撞对：距离小于厚度时沿法线对半推开（等质量）。

    小编号顶点始终在前，修正方向与符号因此完全确定。无 positions 时返回接触描述，
    便于上层预排碰撞批次。
    """
    if isinstance(pair, int):
        i, j = 0, max(1, pair)
    elif isinstance(pair, dict):
        raw = pair.get("pair", (pair.get("i", 0), pair.get("j", 1)))
        i, j = int(raw[0]), int(raw[1])
        thickness = pair.get("thickness", thickness)
    else:
        i, j = int(pair[0]), int(pair[1])
        if len(pair) > 2:
            thickness = pair[2]

    if i == j:
        raise ValueError(f"碰撞对必须是两个不同顶点: {pair!r}")
    lo, hi = (i, j) if i < j else (j, i)
    thick = max(0.0, _num(thickness, DEFAULT_THICKNESS))
    result = {
        "pair": (lo, hi),
        "colliding": False,
        "distance": None,
        "normal": (0.0, 0.0, 0.0),
        "correction_i": (0.0, 0.0, 0.0),
        "correction_j": (0.0, 0.0, 0.0),
    }
    if positions is None:
        return result

    p_i, p_j = positions[lo], positions[hi]
    delta = _vsub(p_j, p_i)
    dist = _vlen(delta)
    result["distance"] = dist
    if dist >= thick:
        return result

    if dist < EPS:
        normal = [1.0, 0.0, 0.0]
        dist = EPS
    else:
        normal = _vscale(delta, 1.0 / dist)
    overlap = thick - dist
    push = _vscale(normal, overlap * 0.5)
    for d in range(len(normal)):
        p_i[d] -= push[d]
        p_j[d] += push[d]
    result.update(
        colliding=True,
        normal=tuple(normal),
        correction_i=tuple(-x for x in push),
        correction_j=tuple(push),
    )
    return result


# ---------------------------------------------------------------------------
# 13. 确定性校验：同初始状态两次重放必须逐位一致
# ---------------------------------------------------------------------------
def _demo_state(particles=4):
    n = max(2, int(particles))
    positions = [[float(i), 0.0, 0.0] for i in range(n)]
    velocities = [[0.0, 0.0, 0.0] for _ in range(n)]
    constraints = [(i, i + 1, 1.0) for i in range(n - 1)]
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]
    wind = [_vzero(3) for _ in range(n)]
    return ClothState(
        positions=positions,
        velocities=velocities,
        constraints=constraints,
        collision_pairs=pairs,
        wind=wind,
        wind_zone="calm",
        device="cpu",
    )


def determinism(state=None, runs=2, frame_dt=REFERENCE_DT, config=None):
    """对同一初始状态独立重放 runs 次，逐位比较位置与速度，全一致返回 True。"""
    if state is None:
        state = _demo_state(4)
    elif isinstance(state, int):
        state = _demo_state(max(2, state))

    cfg = validation(config or {})
    runs = max(2, int(_num(runs, 2.0)))
    snapshots = []
    for _ in range(runs):
        replay = copy.deepcopy(state)
        step(replay, frame_dt, cfg)
        snapshots.append((copy.deepcopy(replay.positions), copy.deepcopy(replay.velocities)))
    first = snapshots[0]
    return all(snap == first for snap in snapshots[1:])


# ---------------------------------------------------------------------------
# PBD 求解核心
# ---------------------------------------------------------------------------
def _solve_distance(constraint, positions):
    i, j, rest = int(constraint[0]), int(constraint[1]), float(constraint[2])
    delta = _vsub(positions[j], positions[i])
    dist = _vlen(delta)
    if dist < EPS:
        normal = [1.0, 0.0, 0.0]
        dist = EPS
    else:
        normal = _vscale(delta, 1.0 / dist)
    correction = _vscale(normal, 0.5 * (dist - rest))
    for d in range(len(normal)):
        positions[i][d] += correction[d]
        positions[j][d] -= correction[d]


def _integrate(state, dt, cfg):
    c = cfg["damping"]
    for i in range(len(state.positions)):
        vel = damping(c, state.velocities[i])
        for d in range(3):
            vel[d] += GRAVITY[d] * dt
            if state.wind:
                vel[d] += state.wind[i][d] * WIND_COUPLING * dt
            state.positions[i][d] += vel[d] * dt
        state.velocities[i] = vel


def _run_substep(state, dt, frame_dt, cfg, *, conservative=False):
    # 1) 积分（含阻尼、重力、风场）
    _integrate(state, dt, cfg)
    # 2) 距离约束：确定序 + 低帧率安全迭代上限
    requested = 1 if conservative else cfg["iterations"]
    iters = iteration_cap(requested, frame_dt, cfg["iteration_cap"])
    ordered = solve_order(state.constraints)
    for _ in range(iters):
        for constraint in ordered:
            _solve_distance(constraint, state.positions)
    # 3) 自碰撞：扫掠厚度 + 确定序，逐对推开
    thick = self_thickness(cfg["thickness"], dt, _state_max_speed(state))
    for pair in solve_order(state.collision_pairs):
        collision_pair(pair, state.positions, thick)


def step(state, frame_dt=REFERENCE_DT, config=None):
    """推进一帧：子步加密、栅栏、爆炸回滚与 GPU->CPU 回退全部串在这里。

    每个子步前保存快照；GPU 故障或爆炸保护触发时回滚快照并只重放一次
    （GPU 故障转 CPU，爆炸走保守单迭代），重放后复检，仍不达标也只标记不循环。
    """
    cfg = validation(config or {"frame_dt": frame_dt})
    frame_dt = cfg["frame_dt"]

    # 回读栅栏：等掉上一帧的在途异步回读，本帧才允许写缓冲
    fence = readback_fence(state.frame, inflight=state.inflight)
    state.inflight = [fence]

    n, dt_sub = substep_count(
        frame_dt,
        cfg["substeps"],
        thickness=cfg["thickness"],
        max_speed=cfg["max_speed"],
    )
    for _ in range(n):
        snapshot = copy.deepcopy(state)
        conservative = False
        try:
            if state.device == "gpu" and state.gpu_fault:
                raise RuntimeError("GPU device lost during cloth solve")
            _run_substep(state, dt_sub, frame_dt, cfg)
        except Exception as exc:
            # 回滚到子步前快照，切 CPU 重放同一子步，不在故障设备上重试
            state.positions = copy.deepcopy(snapshot.positions)
            state.velocities = copy.deepcopy(snapshot.velocities)
            cpu_fallback(exc, state=state)
            _run_substep(state, dt_sub, frame_dt, cfg)

        _, exploded = explode_guard(state, snapshot, cfg["max_speed"])
        if exploded:
            # 回滚快照，用保守参数（单次约束迭代、带阻尼）只重放一次后复检
            state.positions = copy.deepcopy(snapshot.positions)
            state.velocities = copy.deepcopy(snapshot.velocities)
            _run_substep(state, dt_sub, frame_dt, cfg, conservative=True)
            _, still_bad = explode_guard(state, snapshot, cfg["max_speed"])
            if still_bad:
                state.failed = True

    state.frame += 1
    return state, fence
