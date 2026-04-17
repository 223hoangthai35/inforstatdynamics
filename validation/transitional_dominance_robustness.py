"""
T4 -- Transitional Dominance Robustness Check (VNINDEX-only).

Reframed from the original T4 spec: rather than re-testing the rejected
"VN flip rate > SPX flip rate" hypothesis, this script tests whether the
Transitional Dominance pattern reported in Section 5.6.4 is a structural
property of VNINDEX or an artifact of the specific hysteresis parameters
the filter was calibrated to.

Method
------
  1. Load VNINDEX OHLCV once, build Plane-1 features once, fit the GMM
     once. (Features depend only on pinned WPE / SampEn / SPE_Z params;
     the GMM fit depends only on the feature matrix. Both are config-
     independent.)
  2. Apply three HysteresisGMMWrapper configs to the same fitted GMM:
        Config A (current):  delta_hard=0.60, delta_soft=0.35, t_persist=8
        Config B (looser):   delta_hard=0.50, delta_soft=0.30, t_persist=6
        Config C (tighter):  delta_hard=0.70, delta_soft=0.40, t_persist=10
  3. For each: report flip_rate/yr, p(Det/Tra/Sto), T_det/T_tra/T_sto.

Pre-stated predictions
----------------------
  Real structural property -> p(Tra) stable in 60-75% across all configs;
                              T_tra stable in 45-80 days.
  Hysteresis artifact      -> p(Tra) swings dramatically (drops < 50% on
                              looser, climbs > 85% on tighter).

We report ALL three configs honestly regardless of outcome.

Run:
    python validation/transitional_dominance_robustness.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.ds_skill import HysteresisGMMWrapper, EntropyPhaseSpaceClassifier
from validation._features import (
    load_ohlcv, build_plane1_features, TRADING_DAYS,
)
from validation.regime_duration import regime_duration_stats, _slice_common, COMMON_END


CONFIGS: Dict[str, dict] = {
    "A_current": dict(delta_hard=0.60, delta_soft=0.35, t_persist=8),
    "B_looser":  dict(delta_hard=0.50, delta_soft=0.30, t_persist=6),
    "C_tighter": dict(delta_hard=0.70, delta_soft=0.40, t_persist=10),
}

MARKET = "VNINDEX"
TICKER = "VNINDEX"
SOURCE = "vnstock"
DATA_START = "2020-01-01"
DATA_END = COMMON_END
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH = os.path.join(RESULTS_DIR, "transitional_dominance_robustness.json")


def _flip_rate_per_year(labels: pd.Series) -> float:
    arr = labels.astype(int).values
    if len(arr) < 2:
        return 0.0
    return float((np.diff(arr) != 0).sum()) * TRADING_DAYS / len(arr)


def main() -> int:
    print(f"  Loading {MARKET} OHLCV {DATA_START} -> {DATA_END} ...")
    df = load_ohlcv(MARKET, TICKER, SOURCE, DATA_START, DATA_END)
    print(f"  Building Plane-1 features ...")
    feat = build_plane1_features(df)
    print(f"  {len(feat)} labelable bars after rolling SPE_Z.")

    print(f"  Fitting Plane-1 GMM (random_state=42) ...")
    clf = EntropyPhaseSpaceClassifier(n_components=3, random_state=42)
    clf.fit_predict(feat.values)

    rows: List[dict] = []
    for cfg_name, cfg in CONFIGS.items():
        wrapper = HysteresisGMMWrapper(clf, **cfg)
        filt = wrapper.transform(feat.values)
        labels = pd.Series(filt, index=feat.index, name=f"filt_{cfg_name}")
        labels_common = _slice_common(labels)

        stats = regime_duration_stats(labels_common, MARKET)
        fpy = _flip_rate_per_year(labels_common)

        rows.append({
            "config_name": cfg_name,
            "config": cfg,
            "n_bars_common": stats.n_bars,
            "n_segments": stats.n_segments,
            "flips_per_year": fpy,
            "p_det": stats.label_share[0],
            "p_tra": stats.label_share[1],
            "p_sto": stats.label_share[2],
            "T_det_days": stats.by_regime_mean_days[0],
            "T_tra_days": stats.by_regime_mean_days[1],
            "T_sto_days": stats.by_regime_mean_days[2],
            "overall_mean_days": stats.overall_mean_days,
        })

    # ----- print -----
    print()
    print("=" * 96)
    print(f"T4 -- TRANSITIONAL DOMINANCE ROBUSTNESS  (VNINDEX, common window 2022-01-01 -> {COMMON_END})")
    print("=" * 96)
    header = (
        f"{'Config':<12} {'flips/yr':>9}  "
        f"{'p(Det)':>7} {'p(Tra)':>7} {'p(Sto)':>7}  "
        f"{'T_det':>7} {'T_tra':>7} {'T_sto':>7}  {'overall':>8}"
    )
    print(header)
    print("-" * 96)
    for r in rows:
        print(
            f"{r['config_name']:<12} {r['flips_per_year']:>9.2f}  "
            f"{r['p_det']:>7.1%} {r['p_tra']:>7.1%} {r['p_sto']:>7.1%}  "
            f"{r['T_det_days']:>7.2f} {r['T_tra_days']:>7.2f} {r['T_sto_days']:>7.2f}  "
            f"{r['overall_mean_days']:>8.2f}"
        )
    print("-" * 96)
    print("Configs:")
    for cfg_name, cfg in CONFIGS.items():
        print(
            f"  {cfg_name}: delta_hard={cfg['delta_hard']}, "
            f"delta_soft={cfg['delta_soft']}, t_persist={cfg['t_persist']}"
        )

    # ----- save -----
    p_tras = [r["p_tra"] for r in rows]
    t_tras = [r["T_tra_days"] for r in rows]
    payload = {
        "market": MARKET,
        "data_window": [DATA_START, DATA_END],
        "common_window_start": "2022-01-01",
        "common_window_end": COMMON_END,
        "configs": CONFIGS,
        "rows": rows,
        "summary": {
            "p_tra_min": float(min(p_tras)),
            "p_tra_max": float(max(p_tras)),
            "p_tra_range": float(max(p_tras) - min(p_tras)),
            "T_tra_min_days": float(min(t_tras)),
            "T_tra_max_days": float(max(t_tras)),
            "T_tra_range_days": float(max(t_tras) - min(t_tras)),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Result saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
