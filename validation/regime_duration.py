"""
Regime-duration analysis -- discriminator for the three Case A interpretations.

Context
=======
T2 (cross_market_flip_rate.py) found that VNINDEX flips LESS than SPX/BTC,
the opposite of the originally predicted direction. Three competing
interpretations remain:

  (A) Calibration confound  -- hysteresis tuned on VNINDEX, so the lower flip
                               count is an artifact of where the thresholds
                               were set, not a real microstructure property.
  (B) Monetary stickiness   -- SBV intervenes to suppress regime change; VN
                               regimes are "frozen" by policy, not by
                               internal coordination.
  (C) Sustained coordination -- frontier markets retain coordination longer
                               than developed/crypto markets because they
                               lack the institutional arbitrage that breaks
                               coordination in real time. Regimes persist.

Predictions
-----------
  Mean duration (deterministic regime), T_det:
      (A) VN ~ SPX ~ BTC  (durations comparable, only transition timing differs)
      (B) VN >> SPX, VN >> BTC, especially during SBV-active windows
      (C) VN > SPX > BTC  (monotone with market depth / arbitrage capacity)

  Stochastic share, P(label = 2):
      (A) Comparable across markets
      (B) Suppressed in VN
      (C) Suppressed in VN (similar prediction to B)

  Det/Sto duration ratio, T_det / T_sto:
      (B) and (C) both predict elevated, but (C) predicts the same monotone
      pattern in SPX/BTC, while (B) predicts VN as an outlier.

So the (B) vs (C) split rests on the *cross-market gradient* and on whether
non-VN markets show the same ordering. (A) is rejected if any duration
metric differs by >1.5x between VN and SPX.

Run:
    python validation/regime_duration.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.extract_flip_dates import get_filtered_labels


COMMON_START = "2022-01-01"
COMMON_END = "2026-04-17"
NATIVE_BPY = {"VNINDEX": 252, "SP500": 252, "BTC": 365}

LABEL_NAMES = {0: "Deterministic", 1: "Transitional", 2: "Stochastic"}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH = os.path.join(RESULTS_DIR, "regime_duration_result.json")


@dataclass
class DurationStats:
    market: str
    n_bars: int
    n_segments: int
    bars_per_year: int
    overall_mean_bars: float
    overall_mean_days: float
    by_regime_mean_bars: Dict[int, float]
    by_regime_mean_days: Dict[int, float]
    by_regime_count: Dict[int, int]
    label_share: Dict[int, float]
    det_over_sto_duration: float


def _runs(arr: np.ndarray) -> List[tuple]:
    """Return list of (label, run_length) for consecutive-equal runs."""
    if len(arr) == 0:
        return []
    changes = np.flatnonzero(np.diff(arr) != 0) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(arr)]])
    return [(int(arr[s]), int(e - s)) for s, e in zip(starts, ends)]


def regime_duration_stats(labels: pd.Series, market: str) -> DurationStats:
    arr = labels.astype(int).values
    n_bars = len(arr)
    bpy = NATIVE_BPY[market]
    runs = _runs(arr)
    n_segments = len(runs)
    overall_mean = float(np.mean([r[1] for r in runs])) if runs else 0.0

    by_regime_runs: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for lbl, length in runs:
        by_regime_runs.setdefault(lbl, []).append(length)

    by_regime_mean_bars = {
        lbl: (float(np.mean(v)) if v else 0.0)
        for lbl, v in by_regime_runs.items()
    }
    by_regime_mean_days = {
        lbl: bars * (365.0 / bpy) for lbl, bars in by_regime_mean_bars.items()
    }
    by_regime_count = {lbl: len(v) for lbl, v in by_regime_runs.items()}

    unique, counts = np.unique(arr, return_counts=True)
    share = {int(u): float(c) / n_bars for u, c in zip(unique, counts)}
    for k in (0, 1, 2):
        share.setdefault(k, 0.0)

    sto = by_regime_mean_bars.get(2, 0.0)
    det_over_sto = (
        by_regime_mean_bars.get(0, 0.0) / sto if sto > 0 else float("inf")
    )

    return DurationStats(
        market=market,
        n_bars=n_bars,
        n_segments=n_segments,
        bars_per_year=bpy,
        overall_mean_bars=overall_mean,
        overall_mean_days=overall_mean * (365.0 / bpy),
        by_regime_mean_bars=by_regime_mean_bars,
        by_regime_mean_days=by_regime_mean_days,
        by_regime_count=by_regime_count,
        label_share=share,
        det_over_sto_duration=det_over_sto,
    )


def _slice_common(labels: pd.Series) -> pd.Series:
    s = labels.copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[(s.index >= COMMON_START) & (s.index <= COMMON_END)]


def _format(stats: List[DurationStats]) -> str:
    lines = [
        "=" * 86,
        f"REGIME DURATION (common window {COMMON_START} -> {COMMON_END})",
        "=" * 86,
        f"{'Market':<10} {'n_bars':>7} {'n_seg':>6} {'overall_d':>10}  "
        f"{'T_det_d':>9} {'T_tra_d':>9} {'T_sto_d':>9}  "
        f"{'p(0)':>5} {'p(1)':>5} {'p(2)':>5}  {'Tdet/Tsto':>10}",
        "-" * 86,
    ]
    for s in stats:
        lines.append(
            f"{s.market:<10} {s.n_bars:>7d} {s.n_segments:>6d} "
            f"{s.overall_mean_days:>10.2f}  "
            f"{s.by_regime_mean_days[0]:>9.2f} "
            f"{s.by_regime_mean_days[1]:>9.2f} "
            f"{s.by_regime_mean_days[2]:>9.2f}  "
            f"{s.label_share[0]:>5.1%} "
            f"{s.label_share[1]:>5.1%} "
            f"{s.label_share[2]:>5.1%}  "
            f"{s.det_over_sto_duration:>10.2f}"
        )
    lines += [
        "-" * 86,
        "Legend: T_xxx_d = mean duration of regime xxx in CALENDAR days.",
        "        Det = Deterministic (label 0); Tra = Transitional (1); Sto = Stochastic (2).",
        "",
        "Predictions:",
        "  (A) Calibration confound      -> all overall_d comparable (within 1.5x)",
        "  (B) Monetary stickiness       -> VN T_det >> SPX T_det, VN P(2) suppressed",
        "  (C) Sustained coordination    -> monotone VN > SPX > BTC on T_det",
    ]
    return "\n".join(lines)


def main() -> int:
    markets = ["VNINDEX", "SP500", "BTC"]
    stats: List[DurationStats] = []
    for mkt in markets:
        print(f"  Loading {mkt} ...")
        labels = get_filtered_labels(market=mkt, start="2020-01-01", end=COMMON_END)
        labels = _slice_common(labels)
        if labels.empty:
            print(f"    {mkt}: no labels in common window — skipping")
            continue
        stats.append(regime_duration_stats(labels, mkt))

    print()
    print(_format(stats))

    payload = {
        "common_start": COMMON_START,
        "common_end": COMMON_END,
        "native_bpy": NATIVE_BPY,
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "stats": [
            {
                **asdict(s),
                "by_regime_mean_bars": {str(k): v for k, v in s.by_regime_mean_bars.items()},
                "by_regime_mean_days": {str(k): v for k, v in s.by_regime_mean_days.items()},
                "by_regime_count": {str(k): v for k, v in s.by_regime_count.items()},
                "label_share": {str(k): v for k, v in s.label_share.items()},
            }
            for s in stats
        ],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n  Result saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
