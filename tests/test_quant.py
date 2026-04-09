"""
Unit tests for skills/quant_skill.py

Run with:
    python -m pytest tests/test_quant.py -v
or standalone:
    python tests/test_quant.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.quant_skill import calc_rolling_wpe


# ── WPE bounds ─────────────────────────────────────────────────────────────────

def test_wpe_bounds():
    """WPE must stay in [0, 1] for any finite input."""
    rng = np.random.default_rng(42)
    log_rets = rng.standard_normal(200)
    wpe_arr, _ = calc_rolling_wpe(log_rets, m=3, tau=1, window=22)

    valid = wpe_arr[~np.isnan(wpe_arr)]
    assert len(valid) > 0, "No valid WPE values computed"
    assert float(valid.min()) >= 0.0, f"WPE below 0: {valid.min()}"
    assert float(valid.max()) <= 1.0, f"WPE above 1: {valid.max()}"


# ── Deterministic signal → low WPE ────────────────────────────────────────────

def test_wpe_deterministic_low():
    """A perfectly monotone series (one ordinal pattern) should yield WPE near 0."""
    # Strictly increasing returns → only one ordinal pattern possible
    log_rets = np.linspace(0.001, 0.010, 200)
    wpe_arr, _ = calc_rolling_wpe(log_rets, m=3, tau=1, window=22)

    valid = wpe_arr[~np.isnan(wpe_arr)]
    assert len(valid) > 0

    # Allow some numerical noise — threshold of 0.15 is generous
    assert float(valid.mean()) < 0.15, (
        f"Deterministic signal should produce low WPE, got mean={valid.mean():.4f}"
    )


# ── Random signal → high WPE ──────────────────────────────────────────────────

def test_wpe_random_high():
    """IID Gaussian noise (maximum disorder) should yield WPE near 1."""
    rng = np.random.default_rng(0)
    log_rets = rng.standard_normal(500)
    wpe_arr, _ = calc_rolling_wpe(log_rets, m=3, tau=1, window=22)

    valid = wpe_arr[~np.isnan(wpe_arr)]
    assert len(valid) > 0

    # For m=3 IID noise, WPE converges close to 1.0
    assert float(valid.mean()) > 0.85, (
        f"Random signal should produce high WPE, got mean={valid.mean():.4f}"
    )


# ── NaN handling ──────────────────────────────────────────────────────────────

def test_wpe_nan_prefix():
    """First (window-1) values should be NaN (insufficient history)."""
    log_rets = np.random.default_rng(7).standard_normal(100)
    window = 22
    wpe_arr, _ = calc_rolling_wpe(log_rets, m=3, tau=1, window=window)

    # The first window-1 entries must be NaN
    prefix = wpe_arr[:window - 1]
    assert np.all(np.isnan(prefix)), (
        f"Expected NaN prefix of length {window-1}, "
        f"got {np.sum(~np.isnan(prefix))} non-NaN values"
    )


# ── Ordering invariant: random > deterministic ────────────────────────────────

def test_wpe_ordering():
    """Mean WPE of random series must exceed mean WPE of deterministic series."""
    rng = np.random.default_rng(99)
    rand_rets = rng.standard_normal(300)
    det_rets  = np.linspace(0.001, 0.010, 300)

    wpe_rand, _ = calc_rolling_wpe(rand_rets, m=3, tau=1, window=22)
    wpe_det,  _ = calc_rolling_wpe(det_rets,  m=3, tau=1, window=22)

    mean_rand = float(np.nanmean(wpe_rand))
    mean_det  = float(np.nanmean(wpe_det))

    assert mean_rand > mean_det, (
        f"Expected WPE(random) > WPE(deterministic), "
        f"got {mean_rand:.4f} vs {mean_det:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_wpe_bounds,
        test_wpe_deterministic_low,
        test_wpe_random_high,
        test_wpe_nan_prefix,
        test_wpe_ordering,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
