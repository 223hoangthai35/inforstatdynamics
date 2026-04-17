"""
Event Study -- Test 1 of Case A Validation.

H1: filtered regime flips on VNINDEX 2020-2026 are clustered around documented
    macro events at a rate that materially exceeds the random-baseline null.
H0: flips are randomly distributed in time; any apparent clustering is chance.

Method:
  1. Take the post-hysteresis flip dates from the production pipeline.
  2. For each flip, find the nearest pre-registered event (calendar days).
     A flip "matches" an event if it lies within +/- TOLERANCE_DAYS.
  3. Precision = matched_flips / total_flips.
  4. Bootstrap null: sample N random flip dates uniformly across the data
     range, compute the same precision, repeat 10000 times. The p-value is
     the fraction of null samples whose precision matches or exceeds the
     observed value.

PRE-REGISTRATION
================
The KNOWN_EVENTS_VNINDEX dictionary below was compiled from publicly
documented Vietnam-relevant macro events between 2020-01-01 and 2026-04-17,
sourced from contemporaneous reporting by VnExpress, Reuters, the SBV, and
the Federal Reserve. The list was committed to git BEFORE this test was run
against any flip output. Modifying this list after observing the matched /
unmatched split would constitute HARKing -- if a missed event is identified
post hoc, log it in `validation/results/events_discovered_posthoc.md` with
the date and reason, do NOT silently extend the main list.

Run:
    python validation/event_study.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.extract_flip_dates import get_filtered_flip_dates


# ==============================================================================
# PRE-REGISTERED EVENT LIST -- DO NOT MODIFY AFTER FIRST COMMIT
# ==============================================================================
# Format: 'YYYY-MM-DD': ('event_name', 'category')
# Categories used: pandemic | bubble | geopolitical | credit_event | macro_policy
KNOWN_EVENTS_VNINDEX: Dict[str, tuple] = {
    # --- 2020 COVID cycle ---
    '2020-01-23': ('COVID first reported in Vietnam',           'pandemic'),
    '2020-03-12': ('WHO declares pandemic',                     'pandemic'),
    '2020-03-24': ('VNINDEX panic low / global circuit-breaker week', 'pandemic'),
    '2020-04-01': ('Vietnam national lockdown begins',          'pandemic'),
    # --- 2021 retail bubble / Delta wave ---
    '2021-01-19': ('VNINDEX breaks all-time high; retail boom', 'bubble'),
    '2021-07-12': ('Delta-variant lockdown in Ho Chi Minh City', 'pandemic'),
    # --- 2022 bear / credit events ---
    '2022-01-04': ('VNINDEX peak (~1530) -- bubble top',        'bubble'),
    '2022-02-24': ('Russia invades Ukraine',                    'geopolitical'),
    '2022-04-05': ('Tan Hoang Minh chairman arrested',          'credit_event'),
    '2022-05-05': ('FLC chairman arrested',                     'credit_event'),
    '2022-10-07': ('Van Thinh Phat / SCB crisis breaks',        'credit_event'),
    '2022-11-15': ('Corporate-bond market freeze',              'credit_event'),
    # --- 2023 global credit shock ---
    '2023-03-10': ('SVB collapse',                              'credit_event'),
    '2023-03-15': ('Credit Suisse forced merger',               'credit_event'),
    # --- 2024 macro policy ---
    '2024-04-01': ('Fed rate-cut expectation reset',            'macro_policy'),
    '2024-09-18': ('Fed cuts 50 bp',                            'macro_policy'),
    # --- 2025 tariff regime ---
    '2025-01-20': ('Trump inauguration / tariff uncertainty',   'geopolitical'),
    '2025-04-02': ('"Liberation Day" tariffs announced',        'geopolitical'),
}

TOLERANCE_DAYS: int = 10           # +/- calendar days for a flip-event match
N_BOOTSTRAP:    int = 10_000
RANDOM_SEED:    int = 42

PRECISION_THRESHOLD = 0.60         # T1 PASS iff precision > this AND p < 0.05
PVALUE_THRESHOLD    = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH = os.path.join(RESULTS_DIR, "event_study_result.json")


# ==============================================================================
# RESULT DATACLASS
# ==============================================================================
@dataclass
class EventStudyResult:
    market: str
    start: str
    end: str
    total_flips: int
    matched_flips: int
    precision: float
    unmatched_flip_dates: List[pd.Timestamp]
    matched_events: Dict[pd.Timestamp, pd.Timestamp]   # event -> flip
    p_value_vs_random: float
    null_mean_precision: float
    null_p95_precision: float
    tolerance_days: int = TOLERANCE_DAYS
    n_bootstrap: int = N_BOOTSTRAP
    seed: int = RANDOM_SEED
    event_list_size: int = field(default_factory=lambda: len(KNOWN_EVENTS_VNINDEX))


# ==============================================================================
# CORE TEST
# ==============================================================================
def _precision_for_random_sample(
    rng: np.random.Generator,
    n_flips: int,
    business_days_int: np.ndarray,
    event_int_sorted: np.ndarray,
    tolerance: int,
) -> float:
    """One bootstrap iteration -- precision of n_flips uniformly random dates."""
    sample = rng.choice(business_days_int, size=n_flips, replace=False)
    # Vectorized nearest-event distance via searchsorted on sorted events.
    idx = np.searchsorted(event_int_sorted, sample)
    left = np.clip(idx - 1, 0, len(event_int_sorted) - 1)
    right = np.clip(idx,     0, len(event_int_sorted) - 1)
    d_left  = np.abs(sample - event_int_sorted[left])
    d_right = np.abs(sample - event_int_sorted[right])
    nearest = np.minimum(d_left, d_right)
    return float((nearest <= tolerance).sum()) / n_flips


def run_event_study(
    flip_dates: List[pd.Timestamp],
    events: Dict[str, tuple] = KNOWN_EVENTS_VNINDEX,
    tolerance_days: int = TOLERANCE_DAYS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    market: str = "VNINDEX",
    start: str = "2020-01-01",
    end: str = "2026-04-17",
) -> EventStudyResult:
    if not flip_dates:
        raise ValueError("No flip dates supplied -- pipeline returned empty.")

    flip_idx = pd.to_datetime(pd.Series(flip_dates)).sort_values().reset_index(drop=True)
    event_idx = pd.to_datetime(pd.Series(list(events.keys()))).sort_values().reset_index(drop=True)

    matched: Dict[pd.Timestamp, pd.Timestamp] = {}
    unmatched: List[pd.Timestamp] = []
    for flip in flip_idx:
        deltas = (event_idx - flip).abs()           # TimedeltaIndex
        i_min = int(deltas.values.argmin())
        if deltas.iloc[i_min].days <= tolerance_days:
            matched[event_idx.iloc[i_min]] = flip
        else:
            unmatched.append(flip)

    precision = len(matched) / len(flip_idx)

    # ------------------------------------------------------------------
    # Bootstrap null distribution: random flip dates uniform on business days.
    # NOTE: pd.Series([Timestamp]).astype('int64') unit is environment-
    # dependent (microseconds in pandas 2.x, nanoseconds in older builds);
    # casting through datetime64[D] makes the day-index conversion explicit
    # and unit-independent.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    biz_days = pd.date_range(flip_idx.iloc[0], flip_idx.iloc[-1], freq="B")
    biz_int = biz_days.values.astype("datetime64[D]").astype("int64")
    event_int = np.sort(
        event_idx.values.astype("datetime64[D]").astype("int64")
    )

    null_precisions = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        null_precisions[i] = _precision_for_random_sample(
            rng, len(flip_idx), biz_int, event_int, tolerance_days
        )
    p_value = float((null_precisions >= precision).mean())

    return EventStudyResult(
        market=market, start=start, end=end,
        total_flips=len(flip_idx),
        matched_flips=len(matched),
        precision=precision,
        unmatched_flip_dates=unmatched,
        matched_events=matched,
        p_value_vs_random=p_value,
        null_mean_precision=float(null_precisions.mean()),
        null_p95_precision=float(np.percentile(null_precisions, 95)),
        tolerance_days=tolerance_days,
        n_bootstrap=n_bootstrap,
        seed=seed,
        event_list_size=len(events),
    )


# ==============================================================================
# REPORTING
# ==============================================================================
def format_result(r: EventStudyResult) -> str:
    verdict = (
        "PASS" if r.precision > PRECISION_THRESHOLD
        and r.p_value_vs_random < PVALUE_THRESHOLD
        else "NEEDS INVESTIGATION"
    )
    lines = [
        "=" * 68,
        f"EVENT STUDY -- {r.market} {r.start} -> {r.end}",
        "=" * 68,
        f"Pre-registered events:    {r.event_list_size}",
        f"Total filtered flips:     {r.total_flips}",
        f"Matched to known events:  {r.matched_flips}",
        f"Precision:                {r.precision:.1%}",
        f"  null mean:              {r.null_mean_precision:.1%}",
        f"  null 95th pct:          {r.null_p95_precision:.1%}",
        f"  p-value:                {r.p_value_vs_random:.4f}",
        f"Tolerance:                +/- {r.tolerance_days} calendar days",
        f"Bootstrap iterations:     {r.n_bootstrap}",
        "",
        f"VERDICT: {verdict}",
        "",
        "MATCHED  (event -> flip, signed delta in calendar days):",
    ]
    for evt, flip in sorted(r.matched_events.items()):
        delta = (flip - evt).days
        evt_meta = KNOWN_EVENTS_VNINDEX.get(evt.strftime('%Y-%m-%d'), ('', ''))
        lines.append(
            f"  {evt.strftime('%Y-%m-%d')} -> {flip.strftime('%Y-%m-%d')} "
            f"({delta:+d}d)  {evt_meta[0]}"
        )
    if r.unmatched_flip_dates:
        lines.append("")
        lines.append("UNMATCHED FLIPS (investigate or log post-hoc):")
        for flip in r.unmatched_flip_dates:
            lines.append(f"  {flip.strftime('%Y-%m-%d')}")
    return "\n".join(lines)


def save_result(r: EventStudyResult, path: str = RESULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "market": r.market,
        "start": r.start,
        "end": r.end,
        "total_flips": r.total_flips,
        "matched_flips": r.matched_flips,
        "precision": r.precision,
        "p_value": r.p_value_vs_random,
        "null_mean_precision": r.null_mean_precision,
        "null_p95_precision": r.null_p95_precision,
        "tolerance_days": r.tolerance_days,
        "n_bootstrap": r.n_bootstrap,
        "seed": r.seed,
        "event_list_size": r.event_list_size,
        "precision_threshold": PRECISION_THRESHOLD,
        "p_value_threshold": PVALUE_THRESHOLD,
        "matched": {
            k.strftime("%Y-%m-%d"): v.strftime("%Y-%m-%d")
            for k, v in sorted(r.matched_events.items())
        },
        "unmatched": [d.strftime("%Y-%m-%d") for d in r.unmatched_flip_dates],
        "verdict": (
            "PASS" if r.precision > PRECISION_THRESHOLD
            and r.p_value_vs_random < PVALUE_THRESHOLD
            else "NEEDS INVESTIGATION"
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main() -> int:
    end = "2026-04-17"
    start = "2020-01-01"
    market = "VNINDEX"
    print(f"  Loading filtered flip dates from {market} pipeline ...")
    flips = get_filtered_flip_dates(market=market, start=start, end=end)
    print(f"  Got {len(flips)} filtered flip dates.")
    result = run_event_study(
        flips, market=market, start=start, end=end
    )
    print(format_result(result))
    save_result(result)
    print(f"\n  Result saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
