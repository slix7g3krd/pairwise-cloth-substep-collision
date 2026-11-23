# pairwise-cloth-substep-collision

## 领域

游戏、图形与多媒体

## 主题

布料子步与自碰撞

## 说明

子步厚度、迭代上限与回读栅栏。

初始状态下 `gameplay/cloth_substep_collision.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
