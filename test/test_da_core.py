"""
tests/test_da_core.py
Run with:  python tests/test_da_core.py   (from repo root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from da.core import tempering_schedule, aoei

def test_schedule_sums_to_one():
    for nt in [1,2,3,5,10]:
        for a in [0.0, 0.5, 2.0, 5.0]:
            s = tempering_schedule(nt, a)
            assert len(s) == nt
            assert abs(s.sum() - 1.0) < 1e-5, f"Nt={nt} as={a}: sum={s.sum()}"
    print("PASS  tempering_schedule: sums to 1")

def test_schedule_equal_weights_at_zero():
    for nt in [1, 3, 5, 10]:
        s = tempering_schedule(nt, 0.0)
        assert np.allclose(s, 1.0/nt, atol=1e-5), f"Not equal at as=0: {s}"
    print("PASS  tempering_schedule: equal weights when alpha_s=0")

def test_schedule_back_loaded():
    s = tempering_schedule(5, 2.0)
    assert np.all(np.diff(s) > 0), f"Not back-loaded: {s}"
    print(f"PASS  tempering_schedule: back-loaded  {np.round(s,4)}")

def test_schedule_ntemp1():
    s = tempering_schedule(1, 2.0)
    assert len(s) == 1 and abs(s[0] - 1.0) < 1e-6
    print("PASS  tempering_schedule: Ntemp=1 -> [1.0]")

def test_aoei_floor():
    rng = np.random.default_rng(0)
    for _ in range(100):
        nobs = rng.integers(1, 10)
        Ne   = 15
        R0   = np.abs(rng.standard_normal(nobs)) * 10 + 1.0
        yo   = rng.standard_normal(nobs) * 5
        hxf  = rng.standard_normal((nobs, Ne)) * 3
        r    = aoei(yo, hxf, R0)
        assert np.all(r >= R0 - 1e-6), f"Floor violated: {r} < {R0}"
    print("PASS  aoei: floor guarantee (R_tilde >= R0) for 100 random cases")

def test_aoei_no_inflation_small_dep():
    rng = np.random.default_rng(1)
    Ne = 20; R0 = np.array([25.0])
    yo = np.array([10.0])
    hxf = 9.5 + rng.standard_normal((1, Ne)) * 1.5   # spread~1.5, d~0.5
    r = aoei(yo, hxf, R0)
    assert np.allclose(r, R0, atol=0.01), f"Expected no inflation: {r}"
    print(f"PASS  aoei: no inflation for small departure  R_tilde={r[0]:.3f}")

def test_aoei_inflates_large_dep():
    rng = np.random.default_rng(2)
    Ne = 20; R0 = np.array([25.0])
    yo = np.array([50.0])
    hxf = 10.0 + rng.standard_normal((1, Ne)) * 1.0   # d~40
    r = aoei(yo, hxf, R0)
    assert r[0] > R0[0], f"Expected inflation: {r[0]} <= {R0[0]}"
    print(f"PASS  aoei: inflates large departure  R_tilde={r[0]:.1f}  R0={R0[0]:.1f}")

if __name__ == "__main__":
    print("Running tests\n" + "-"*40)
    test_schedule_sums_to_one()
    test_schedule_equal_weights_at_zero()
    test_schedule_back_loaded()
    test_schedule_ntemp1()
    test_aoei_floor()
    test_aoei_no_inflation_small_dep()
    test_aoei_inflates_large_dep()
    print("-"*40)
    print("All tests passed.")
