import numpy as np

from labloop.oracle.surface import CombinationSurface, HillDrug, Oracle, SynergyPocket

A = HillDrug("A", ec50=1.0, hill=1.2, emax=0.8)
B = HillDrug("B", ec50=0.5, hill=1.0, emax=0.7)
C = HillDrug("C", ec50=2.0, hill=1.0, emax=0.5)


def grid(*drugs):
    d = np.logspace(-2, 1, 7)
    return np.stack(np.meshgrid(*[d] * len(drugs), indexing="ij"), axis=-1)


def test_hill_endpoints():
    assert A.viability(0.0) == 1.0
    assert abs(A.viability(1.0) - (1 - 0.8 / 2)) < 1e-12


def test_no_pocket_is_bliss_independent():
    s = CombinationSurface((A, B, C))
    x = grid(A, B, C)
    expected = A.viability(x[..., 0]) * B.viability(x[..., 1]) * C.viability(x[..., 2])
    assert np.allclose(s.viability(x), expected)
    assert np.allclose(s.true_excess(x), 0.0)


def test_pocket_peaks_at_center_and_needs_its_drugs():
    s = CombinationSurface((A, B, C), (SynergyPocket((0, 1), (0.0, -0.3), 0.2, 0.3),))
    assert abs(s.true_excess(np.array([1.0, 10**-0.3, 0.0])) - 0.3) < 1e-9
    assert s.true_excess(np.array([1.0, 0.0, 1.0])) == 0.0
    assert s.true_excess(np.array([1.0, 10**0.5, 0.0])) < 0.05


def test_viability_stays_in_unit_interval():
    s = CombinationSurface((A, B), (SynergyPocket((0, 1), (1.0, 1.0), 0.3, 5.0),))
    v = s.viability(grid(A, B))
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_oracle_excess_zero_for_singles_and_grows_with_time():
    o = Oracle({"L": CombinationSurface((A, B), (SynergyPocket((0, 1), (0.0, -0.3), 0.2, 0.2),))})
    assert abs(o.true_excess("L", (1.0, 0.0), 48)) < 1e-12
    e24, e72 = (o.true_excess("L", (1.0, 10**-0.3), t) for t in (24, 72))
    assert 0 < e24 < e72
