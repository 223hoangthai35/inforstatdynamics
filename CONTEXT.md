# BỐI CẢNH DỰ ÁN — Financial Entropy Agent

> **Read this file first.** This is the orientation anchor for new Claude
> Code sessions in this repo. The codebase has multiple branches, two
> in-flight paper versions, and a live validation experiment. The Cliff
> Notes are below; deep details live in the linked files.

---

## 1. What this repo is

A statistical-physics surveillance engine for the Vietnamese equity market
(VN-INDEX, VN30) and a comparative cross-market platform (S&P 500, BTC).
The thesis: in Type-2 chaotic systems like financial markets, **low entropy
= coordinated behaviour = fragility**, inverting the gas-thermodynamics
intuition.

  - User-facing description: [README.md](README.md)
  - Technical spec: [architecture.md](architecture.md) (v7.1)
  - Developer rules: [CLAUDE.md](CLAUDE.md)

---

## 2. Branch landscape

| Branch                          | Status   | Purpose                                                              |
|---------------------------------|----------|----------------------------------------------------------------------|
| `master`                        | stable   | v7.0 baseline (pre-hysteresis), production-deployed dashboard        |
| `v7.1-hysteresis-rolling-garch` | merged-ish | adds Schmitt-trigger hysteresis filter + rolling SPE_Z + GARCH-X    |
| `v7.2-case-a-validation`        | **active** | Case A scientific validation of v7.1's filtered flip rate (~7.8/yr) |
| `phase0..phase5-*`              | archived | historical milestone branches; not active                            |

**Default working branch: `v7.2-case-a-validation`** for any work touching
validation, paper v2 drafts, or extending the regime-duration analysis.

---

## 3. Paper versions and current hypotheses

### Paper v1 — Entropy Paradox (published)
Core finding: deterministic-regime return volatility on VNINDEX is HIGHER
than stochastic-regime volatility, the inverse of the SPX pattern. Other
key results referenced as "v1 V*":
  - **V1-V3:** entropy plane structure, regime classification validation
  - **V4:** simple-volatility H-statistic outperforms entropy on overall
    volatility prediction; entropy delivers ~5.5x Lift on tail events
    (entropy's marginal value is at *regime identity*, not regime-average vol)

### Paper v2 — Section 5.6 (drafting on `v7.2-case-a-validation`)
Title: "Regime Persistence under Hysteresis Filtering". Five subsections
covering the Case A validation programme. Draft at
[paper_artifacts/section_5_6_draft.md](paper_artifacts/section_5_6_draft.md).
**Headline new result: Transitional Dominance** (VN spends 67.8% of bars
in Transitional regime with 62-day mean spells, double SPX and triple BTC).
Frame this as a *reformulation* of the flip-rate-as-fragility intuition,
NOT a refutation of v1's Entropy Paradox.

---

## 4. Validation results (Case A, as of 2026-04-17)

| Test                       | Status         | Headline                                                  |
|----------------------------|----------------|-----------------------------------------------------------|
| T1 event study (T1)        | NULL           | precision 16.1% vs null 12.7%, p=0.359                    |
| T1-revised (post-2022)     | NULL           | precision 16.1%, p=0.359                                  |
| T1-wide-tolerance (+-21d)  | NULL (strong)  | precision 19.4%, null mean 24.7%, p=0.814                 |
| T2 cross-market flip rate  | REFORMULATED   | VN/SPX = 0.82x in all panels — opposite of predicted dir  |
| T3 shuffle test            | **PASS**       | observed 7.77 vs null 118.36 flips/yr, p<1e-4 STRUCTURED  |
| T-D regime duration        | NEW PATTERN    | Transitional Dominance falsifies (A)/(B)/(C) interpretations |
| T4 robustness              | **PASS**       | p(Tra) ∈ {66.1%, 67.8%, 68.8%} across 3 hysteresis configs — structural |

Raw JSON results: [validation/results/](validation/results/).

**Pre-registration discipline:** the T1 event list (`KNOWN_EVENTS_VNINDEX`
in [validation/event_study.py](validation/event_study.py)) was committed
in `4b146b1` BEFORE T1 was run. Do not modify it post-hoc; log new event
candidates to [validation/events_discovered_posthoc.md](validation/events_discovered_posthoc.md)
instead. Modifying it would constitute HARKing.

---

## 5. API name vs paper terminology (terms-of-art)

The code and paper use related but non-identical vocabulary. When reading
the paper, translate to:

| Paper term                          | Code symbol / location                                                                     |
|-------------------------------------|--------------------------------------------------------------------------------------------|
| WPE, weighted permutation entropy   | `calc_rolling_wpe()` in [skills/quant_skill.py](skills/quant_skill.py); m=3, tau=1, win=22 |
| SPE_Z, standardised price SampEn    | `calc_rolling_price_sample_entropy()` + `cal_spe_z_rolling()` (win=504), same file         |
| Plane 1                             | `[WPE, SPE_Z]` matrix; built by `validation/_features.build_plane1_features()`             |
| Plane 2 (volume)                    | `[Vol_Shannon, Vol_SampEn]`; PowerTransform + GMM                                          |
| Deterministic (Det) regime          | label=0 in `EntropyPhaseSpaceClassifier`; "Stable" in some older docs                      |
| Transitional regime                 | label=1; "Fragile" in some older docs                                                      |
| Stochastic (Sto) regime             | label=2; "Chaos" in some older docs                                                        |
| Hysteresis filter                   | `HysteresisGMMWrapper` in [skills/ds_skill.py](skills/ds_skill.py)                         |
| Filtered flip rate                  | `flip_rate_per_year(out["filtered_labels"])` in [validation/_features.py](validation/_features.py) |
| Tri-vector composite (v1)           | REMOVED in v7.1; do not re-add. GARCH-X is the sole risk engine post-v7.1.                 |
| Transitional Dominance              | New term, paper v2 only. See Section 5.6.4.                                                |

Old names "Stable / Fragile / Chaos" appear in [CLAUDE.md](CLAUDE.md) and
some docstrings — these map to **Deterministic / Transitional / Stochastic**
respectively in current code (`REGIME_NAMES` in [skills/ds_skill.py](skills/ds_skill.py#L35)).
The old triple is wrong about the danger ordering; the new triple is what
the dashboard ships.

---

## 6. Pinned hyperparameters (research constants — do not parameterise)

These are baked into [validation/_features.py](validation/_features.py) and
mirrored from the production pipeline. Changing any of these invalidates
the entire validation suite (T1, T2, T3, T-D). If you need to sweep them,
do it in a NEW branch under a new name.

| Constant                | Value | File                                              |
|-------------------------|-------|---------------------------------------------------|
| WPE m                   | 3     | [skills/quant_skill.py](skills/quant_skill.py)    |
| WPE tau                 | 1     | same                                              |
| WPE rolling window      | 22    | same                                              |
| Price SampEn window     | 60    | same                                              |
| SPE_Z rolling window    | 504   | same; also note ~2-year labeling floor caveat     |
| GMM n_components        | 3     | [skills/ds_skill.py](skills/ds_skill.py)          |
| GMM covariance_type     | full  | same                                              |
| Hysteresis delta_hard   | 0.60  | `HYSTERESIS_DELTA_HARD` in ds_skill.py            |
| Hysteresis delta_soft   | 0.35  | `HYSTERESIS_DELTA_SOFT`                           |
| Hysteresis t_persist    | 8     | `HYSTERESIS_T_PERSIST`                            |
| Refit interval          | 21    | `REFIT_INTERVAL` (rolling GMM)                    |

Hysteresis was calibrated on VNINDEX post-2020 to a 4-10 flips/yr target
band (achieves ~7.8/yr). Calibration is in
[scripts/calibrate_hysteresis.py](scripts/calibrate_hysteresis.py).

---

## 7. Operational invariants (decisions made; do not re-litigate without evidence)

  - **No PowerTransformer on Plane 1.** Plane 1 GMM operates on raw
    [WPE, SPE_Z]. Plane 2 (volume) DOES use PowerTransformer. Mixing this
    up will produce non-reproducible regime labels.

  - **SPE_Z is rolling, not global, in production.** The global variant
    is retained ONLY for static dashboard scatter overlays. Switching the
    pipeline to global SPE_Z reintroduces look-ahead bias.

  - **GARCH-X is the sole risk engine after v7.1.** The v7.0 tri-vector
    fallback was deliberately removed in commit `d364009`. If GARCH-X
    fails, the dashboard surfaces the failure rather than substituting
    a different score.

  - **Kinematic indicators (V_WPE, a_WPE) are XAI-only.** They are
    computed for LLM narrative generation and never enter GMM input or
    composite score. See CLAUDE.md "XAI Decoupling" section.

  - **`KNOWN_EVENTS_VNINDEX` is frozen post-pre-registration.** Any
    revision invalidates T1's pre-registration claim.

  - **The 504-day SPE_Z window precludes labeling for the first ~2 years
    of any dataset.** All cross-market analysis uses a common window
    starting 2022-01-01. The COVID-19 shock (March 2020) is in the
    unlabelable region — explicitly disclosed in T2 results.

  - **Transitional Dominance is a structural property of VNINDEX.**
    Confirmed by T4 robustness: p(Tra) varies only 2.7 pp across three
    hysteresis configurations spanning loose-to-tight filter
    aggressiveness. Treat p(Tra) ≈ 67% on VNINDEX as a market
    invariant, not a parameter-dependent quantity.

---

## 8. Scripts you will likely need

  - Run dashboard: `streamlit run dashboard.py`
  - Run agent orchestrator: `python agent_orchestrator.py`
  - Recalibrate hysteresis: `python scripts/calibrate_hysteresis.py`
  - Re-run a validation test:
    `python validation/event_study.py` (T1)
    `python validation/event_study_robustness.py` (T1 addenda)
    `python validation/cross_market_flip_rate.py` (T2)
    `python validation/shuffle_test.py` (T3)
    `python validation/regime_duration.py` (T-D)
    `python validation/transitional_dominance_robustness.py` (T4)

All validation scripts memoise the GMM pipeline per `(market, start, end)`
via [scripts/extract_flip_dates.py](scripts/extract_flip_dates.py), so
running multiple tests in the same Python process reuses the fit.

No build step, no test runner, no linter. Each `skills/*.py` file has an
`if __name__ == "__main__":` block for standalone smoke testing.

---

## 9. Where to look when something feels off

  - "Why is the flip rate suddenly different?" → check
    `HysteresisGMMWrapper` parameters. They are immutable defaults; any
    drift means someone overrode them. Also check if SPE_Z accidentally
    flipped to global.
  - "Why does the GMM emit different labels?" → `random_state=42` is
    pinned everywhere. If labels look re-shuffled, check that
    `_label_map` (semantic sort by centroid entropy) is being applied.
  - "Why does the validation script pull empty data?" → vnstock and
    yfinance both have rate limits. Re-run with a sleep, or shrink the
    date range.
  - "Why is the bootstrap p-value suspicious?" → see the unit-conversion
    fix in [validation/event_study.py](validation/event_study.py)
    (`datetime64[D]` casting). pandas builds disagree on the int64
    representation of timestamps.

---

## 10. Conventions for new work

  - Validation tests and addenda go under [validation/](validation/),
    with results saved as JSON in [validation/results/](validation/results/).
  - Paper drafts go under [paper_artifacts/](paper_artifacts/).
  - Discoveries-after-pre-registration go in markdown disclosure logs
    next to the relevant pre-registered artifact (see
    [validation/events_discovered_posthoc.md](validation/events_discovered_posthoc.md)
    for the pattern).
  - Commit messages: imperative mood, scope prefix (`feat(validation):`,
    `fix(hysteresis):`, `docs(paper):`); explain WHY the change was
    necessary, not what the diff says.

---

*Last updated: 2026-04-17. If you change something here that future
sessions need to know, update both this file and the relevant section
in [CLAUDE.md](CLAUDE.md) — they are sister documents (CLAUDE.md is the
developer how-to; CONTEXT.md is the project state-of-play).*
