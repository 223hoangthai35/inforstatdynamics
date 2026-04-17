# Events Discovered Post-Hoc

**Status:** Disclosure log only. Entries here are NOT merged into
`KNOWN_EVENTS_VNINDEX` in [validation/event_study.py](event_study.py); doing so
would be HARKing (Hypothesizing After Results are Known) and would invalidate
the pre-registered T1 event-study test.

## Purpose

T1 returned a null verdict (precision 16.1%, p=0.359). Of the 31 filtered
flips on VNINDEX 2020-2026, 26 fell outside the +/- 10 calendar-day window of
any pre-registered macro event. The clustering pattern of those unmatched
flips is informative — it concentrates in three windows that correspond to
documented Vietnamese-domestic episodes the original event list under-sampled.

Logging them here:
  - preserves the original T1's pre-registration integrity,
  - flags hypotheses for a future paper extension where a fresh, independent
    flip series (e.g. an out-of-sample period) could test against this
    enriched event list,
  - makes the disclosure machine-checkable and auditable.

## Cluster 1 — 2022 Q2-Q3: Property-sector unwind continuation

Unmatched flips: `2022-06-08`, `2022-07-01`, `2022-07-26`, `2022-08-01`.

The pre-registered list covers the headline credit shocks (Tan Hoang Minh
arrest 2022-04-05, FLC arrest 2022-05-05) but not the propagation phase: in
June-August 2022 the Vietnamese property and corporate-bond complex saw
escalating margin calls, forced selling at major brokerages, and SBV
liquidity interventions. Each of these is a candidate stand-alone event.

## Cluster 2 — 2022 Q4 / 2023 Q1: Bond-market freeze aftermath

Unmatched flips: `2022-12-09`, `2022-12-29`, `2023-01-11`, `2023-02-08`,
`2023-02-21`.

The pre-registered list covers Van Thinh Phat (2022-10-07) and the bond
freeze headline (2022-11-15). Not covered: the SBV's Decree 65/Decree 08
sequence on bond restructuring (Dec 2022 / early 2023), Tet liquidity stress,
and the SVB precursor re-pricing in mid-February 2023.

## Cluster 3 — 2023 H2: Pre-SVB-aftermath drift

Unmatched flips: `2023-04-25`, `2023-06-08`, `2023-08-21`, `2023-09-22`,
`2023-12-26`, `2023-12-28`.

This window contains the SBV easing cycle (multiple rate cuts April-June
2023) and the USD/VND devaluation-defense intervention in Aug-Sep 2023, both
of which the pre-registered list under-samples (only generic macro_policy
events are tagged for 2023).

## Cluster 4 — 2024 H2: Domestic bank-stock re-rating

Unmatched flips: `2024-03-14`, `2024-07-16`, `2024-07-22`, `2024-08-06`,
`2024-10-18`, `2024-10-28`.

The pre-registered list captures Fed events (April + September 2024). Missed
candidates: the Vietnamese typhoon Yagi disaster window (early September
2024), the SBV gold-bullion intervention sequence (June-August 2024), and
domestic bank-recapitalization news flow.

## Cluster 5 — 2025 H2 / 2026 Q1: Tariff-aftermath re-equilibration

Unmatched flips: `2025-06-20`, `2025-07-02`, `2025-10-22`, `2025-12-22`,
`2026-01-12`.

Pre-registered list captures the 2025-04-02 "Liberation Day" tariff
announcement. Not covered: the multi-stage tariff-implementation calendar,
EU-Vietnam trade frictions in Q3 2025, and 2026 fiscal policy uncertainty.

## Methodological note

These clusters share a common structural feature: each is a *propagation*
phase of an already-pre-registered shock, not an independent shock. This is
consistent with the "sustained coordination" interpretation of T2 (paper
Section 5.6.4) — frontier-market regimes appear to persist through extended
post-shock equilibration windows rather than reverting to baseline within the
+/- 10-day tolerance the headline event list assumes.

A pre-registered v2 event list for a future independent test should:
  1. Include propagation-window markers (decree announcement dates, bank
     intervention dates), not just headline shock dates.
  2. Either widen the tolerance window to +/- 21 calendar days or use
     event-windows rather than point events for the matching test.
  3. Be committed before any flip series for the new test period is
     extracted from the pipeline.
