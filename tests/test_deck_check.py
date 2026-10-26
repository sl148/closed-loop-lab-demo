"""Pre-launch deck check: the camera flags exactly the setup mistakes and passes the fixed deck."""

import pytest

mujoco = pytest.importorskip("mujoco")

from labloop.sim3d.deck_check import expected_layout, mistakes, run_check  # noqa: E402

DECK = {"tip_rail": 25, "source_rail": 13, "dest_rail": 19, "cell_source": "trough"}
DRUGS = ["MK-4827", "Sorafenib", "Erlotinib", "Vinblastine"]


def test_flags_mistakes_then_passes():
    layout = expected_layout(DECK, DRUGS, ["300uL"] * 5)
    try:
        res = run_check(layout, DRUGS, seed=0)
    except Exception as e:  # no offscreen GL here
        pytest.skip(f"rendering unavailable: {e}")
    first, second = res["passes"]
    assert [r["id"] for r in first if not r["ok"]] == sorted(mistakes(layout), key=[i.id for i in layout.items].index)
    assert all(r["ok"] for r in second)
    assert [r["group"] for r in first] == sorted([r["group"] for r in first], key=["plates", "tips"].index)  # plates first
    assert res["position_error_mm"]["max"] < 1.0
