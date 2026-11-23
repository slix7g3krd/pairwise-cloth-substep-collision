import pytest
import gameplay.cloth_substep_collision as mod

CASES = [
    ("substep", (1)),
    ("self_thickness", (1)),
    ("iteration_cap", (1)),
    ("damping", (1)),
    ("readback_fence", (1)),
    ("wind_reset", (1)),
    ("solve_order", (1)),
    ("cpu_fallback", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("metrics", (1)),
    ("explode_guard", (1)),
    ("collision_pair", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
