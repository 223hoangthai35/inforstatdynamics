"""
Event Study -- Robustness addenda for T1.

Runs two variants of the original T1 event study to probe whether the null
verdict is driven by methodological artifacts rather than absence of signal:

  T1-revised        Restrict events to dates >= 2022-01-01. The 504-day rolling
                    SPE_Z window precludes labeling for the first ~2 years of
                    each dataset, so any pre-2022 event is structurally
                    impossible to match. Eliminates 6 such events.

  T1-wide-tolerance Keep all 18 original events, widen the +/- tolerance from
                    10 to 21 calendar days. Domain rationale: Vietnamese
                    regime transitions historically run multi-week (credit
                    events propagate through the system over 3-4 weeks rather
                    than days).

Both addenda are reported ALONGSIDE -- they do not replace -- the original T1.
Results saved to validation/results/event_study_robustness.json.

Run:
    python validation/event_study_robustness.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.extract_flip_dates import get_filtered_flip_dates
from validation.event_study import (
    KNOWN_EVENTS_VNINDEX,
    PRECISION_THRESHOLD,
    PVALUE_THRESHOLD,
    run_event_study,
    format_result,
)


LABEL_FLOOR_CUTOFF = "2022-01-01"
WIDE_TOLERANCE_DAYS = 21
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH = os.path.join(RESULTS_DIR, "event_study_robustness.json")


def _filter_post_floor(events: Dict[str, tuple], cutoff: str) -> Dict[str, tuple]:
    cutoff_ts = pd.Timestamp(cutoff)
    return {
        d: meta for d, meta in events.items()
        if pd.Timestamp(d) >= cutoff_ts
    }


def _serialize(r) -> dict:
    return {
        "total_flips": r.total_flips,
        "matched_flips": r.matched_flips,
        "precision": r.precision,
        "p_value": r.p_value_vs_random,
        "null_mean_precision": r.null_mean_precision,
        "null_p95_precision": r.null_p95_precision,
        "tolerance_days": r.tolerance_days,
        "event_list_size": r.event_list_size,
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


def main() -> int:
    end = "2026-04-17"
    start = "2020-01-01"
    market = "VNINDEX"

    print(f"  Loading filtered flip dates from {market} pipeline ...")
    flips = get_filtered_flip_dates(market=market, start=start, end=end)
    print(f"  Got {len(flips)} filtered flip dates.\n")

    # ---- T1-revised: post-2022 events only ----
    revised_events = _filter_post_floor(KNOWN_EVENTS_VNINDEX, LABEL_FLOOR_CUTOFF)
    print(
        f"  [T1-revised] Filtered events to >= {LABEL_FLOOR_CUTOFF}: "
        f"{len(revised_events)} of {len(KNOWN_EVENTS_VNINDEX)} retained."
    )
    r_revised = run_event_study(
        flip_dates=flips,
        events=revised_events,
        market=market, start=start, end=end,
    )
    print(format_result(r_revised))
    print()

    # ---- T1-wide-tolerance: all events, +/- 21d ----
    print(
        f"  [T1-wide-tolerance] All {len(KNOWN_EVENTS_VNINDEX)} events, "
        f"tolerance widened to +/- {WIDE_TOLERANCE_DAYS} calendar days."
    )
    r_wide = run_event_study(
        flip_dates=flips,
        events=KNOWN_EVENTS_VNINDEX,
        tolerance_days=WIDE_TOLERANCE_DAYS,
        market=market, start=start, end=end,
    )
    print(format_result(r_wide))
    print()

    payload = {
        "market": market, "start": start, "end": end,
        "label_floor_cutoff": LABEL_FLOOR_CUTOFF,
        "wide_tolerance_days": WIDE_TOLERANCE_DAYS,
        "precision_threshold": PRECISION_THRESHOLD,
        "p_value_threshold": PVALUE_THRESHOLD,
        "T1_revised": _serialize(r_revised),
        "T1_wide_tolerance": _serialize(r_wide),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Result saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
