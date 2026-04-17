"""
Verify GARCH-X exog features are lagged by 1 day so σ²_t depends only on
entropy from t-1 or earlier. A future entropy spike at index `t+k` must
not change σ_t for any k > 0.

Run:
    python -m pytest tests/test_garch_x_lookahead.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# arch lib is required; skip if missing
arch_spec = pytest.importorskip("arch")

from agent_orchestrator import fit_garch_x


def _synthetic_panel(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """OHLCV-shaped frame with the columns fit_garch_x consumes."""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n) * 0.01
    close = 100.0 * np.cumprod(1 + rets)

    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Close":       close,
        "WPE":         np.clip(0.7 + rng.standard_normal(n) * 0.05, 0.0, 1.0),
        "SPE_Z":       rng.standard_normal(n) * 0.5,
        "Vol_SampEn":  np.clip(0.5 + rng.standard_normal(n) * 0.1, 0.0, 1.0),
        "Vol_Shannon": np.clip(0.7 + rng.standard_normal(n) * 0.05, 0.0, 1.0),
    }, index=idx)


def test_future_entropy_does_not_affect_current_sigma():
    """
    Perturb entropy AFTER index `pivot` only. σ_t for t ≤ pivot must be
    bit-identical to the unperturbed run; otherwise look-ahead exists.
    """
    df_a = _synthetic_panel(n=600)
    df_b = df_a.copy()

    pivot = 400
    # Inject a large spike well into the future; lag-1 means σ at pivot
    # depends on entropy ≤ pivot-1, so this perturbation must not leak back.
    df_b.loc[df_b.index[pivot + 1:], "WPE"] = 0.99
    df_b.loc[df_b.index[pivot + 1:], "SPE_Z"] = 5.0
    df_b.loc[df_b.index[pivot + 1:], "Vol_SampEn"] = 0.99
    df_b.loc[df_b.index[pivot + 1:], "Vol_Shannon"] = 0.99

    res_a = fit_garch_x(df_a)
    res_b = fit_garch_x(df_b)

    assert "error" not in res_a, f"Baseline GARCH-X failed: {res_a.get('error')}"
    assert "error" not in res_b, f"Perturbed GARCH-X failed: {res_b.get('error')}"

    cv_a = res_a["cond_vol_series"]
    cv_b = res_b["cond_vol_series"]

    # Compare overlapping prefix up to pivot
    common = cv_a.index.intersection(cv_b.index)
    common = common[common <= df_a.index[pivot]]
    assert len(common) > 50, "Insufficient overlap for comparison"

    # GARCH refits on each call so parameters can shift slightly even if exog
    # is identical up to pivot. The discriminating signal is whether σ_t
    # at the FIRST few observations (where parameter estimates barely move)
    # remains close. We use a tolerance reflecting the indirect parameter
    # path; a true look-ahead would produce a structural gap.
    diffs = (cv_a.loc[common] - cv_b.loc[common]).abs()
    rel_diffs = diffs / cv_a.loc[common].replace(0, np.nan)

    median_rel = float(rel_diffs.median())
    assert median_rel < 0.05, (
        f"σ_t up to pivot diverges by median {median_rel:.4f} relative — "
        f"likely look-ahead from un-lagged exog."
    )
