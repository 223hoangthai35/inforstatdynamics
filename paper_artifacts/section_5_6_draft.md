# 5.6 Regime Persistence under Hysteresis Filtering

> **Draft note for v2 paper extension.** This section reports the outcome of
> the Case A validation programme launched after v7.1's hysteresis filter
> reduced the raw GMM flip rate on VNINDEX from ~28/yr to ~7.8/yr. Case A is
> the question of whether that residual 7.8/yr flip cadence reflects genuine
> regime dynamics or a calibration artifact. The validation comprises three
> pre-registered tests (T1 event study, T2 cross-market, T3 shuffle test)
> plus a duration-based discriminator (T-D) added after T2's outcome
> reframed the question. Frame for the paper: this section *reformulates*
> the flip-rate-as-fragility hypothesis -- it is not a refutation of paper
> v1's Entropy Paradox, which concerns return-distribution stochasticity
> rather than regime-transition cadence.

## 5.6.1 Temporal Structure Confirmed (T3)

The first question is whether the post-hysteresis label sequence on VNINDEX
contains any temporal structure at all, or whether the labels could equally
well have appeared in any order. We compute the observed annualised flip
rate on the filtered label series, then permute the label sequence 10,000
times (preserving the empirical marginal distribution but destroying
temporal structure) and re-compute the flip rate under each permutation.

Result. Observed flip rate on VNINDEX 2022-04 to 2026-04 is **7.77 flips/yr**
on a label distribution of 25.7% Deterministic / 67.8% Transitional /
6.5% Stochastic. The shuffled-null distribution has mean **118.36 flips/yr**
(s.d. 4.65), with a 5th-95th-percentile band of [110.4, 125.7]. The
observed value lies more than twenty standard deviations below the null
mean; the empirical p-value is `p < 1e-4` (no permutation among 10,000
produced a flip rate as low as observed).

Verdict: T3 STRUCTURED. The hypothesis that the labels are exchangeable in
time is decisively rejected. Whatever the post-hysteresis label series
encodes, it is not noise -- consecutive bars carry information about each
other beyond their unconditional marginals.

## 5.6.2 Cross-Market Flip Rate (T2) -- Unexpected Direction

We next ask whether the VNINDEX flip cadence is anomalously low compared
to a developed-market benchmark (S&P 500) and a 24/7 high-velocity venue
(BTC-USD). The original prediction underpinning the v7.1 calibration was
that VNINDEX, as a frontier market with thin order books and concentrated
retail flow, would show MORE regime-transitions per year than developed
markets, in line with the "frontier fragility" intuition.

We refit the same GMM + hysteresis pipeline on each market over a common
labelable window 2022-01-01 to 2026-04-17 (the 504-bar SPE_Z rolling
window precludes labeling the first ~2 years of each dataset). Three
panels are reported:

| Panel              | Notes                                  | VNINDEX | SP500 | BTC  | Ordering             |
|--------------------|----------------------------------------|--------:|------:|-----:|----------------------|
| HEADLINE           | native calendars (BTC=365, eq=252)     |    7.77 |  9.42 |11.88 | BTC > SPX > VN       |
| ROBUSTNESS-1       | equity-equivalent (all x252)           |    7.77 |  9.42 | 8.20 | SPX > BTC > VN       |
| ROBUSTNESS-2       | full per-market labelable window       |    7.77 |  9.42 |11.78 | BTC > SPX > VN       |

VNINDEX has the LOWEST flip rate in all three panels. The VN/SPX ratio is
0.82x regardless of normalisation (driven by counts on aligned windows).
The originally-predicted direction of inequality is therefore not
supported by the data. *We emphasise this is a reformulation, not a
refutation of paper v1's Entropy Paradox: paper v1 concerns
return-distribution stochasticity (Det vol vs Sto vol per market), not the
cross-market ordering of regime-transition cadences.*

> **Disclosure (COVID window):** The 504-day rolling SPE_Z window
> precludes labeling for approximately the first two years of each
> dataset. The COVID-19 shock (March 2020) falls in the unlabelable
> region. Cross-market headline comparison covers 2022-01-01 onwards,
> spanning Russia-Ukraine war onset, global credit tightening (SVB,
> Credit Suisse), and Vietnamese domestic credit events (Tan Hoang Minh,
> bond market freeze) -- a range of macro regimes sufficient to test
> cross-market microstructure dependency.

## 5.6.3 Event Alignment (T1) and Robustness Addenda

The third pre-registered test asks whether the 31 filtered flips on
VNINDEX 2020-2026 cluster around a list of 18 publicly documented
macro events (pandemic shocks, credit events, Fed policy turns, tariff
announcements). The event list was committed to git BEFORE flip
extraction (commit `4b146b1`); modifying it post-hoc would constitute
HARKing. A flip "matches" an event if it lies within +/- 10 calendar
days; a permutation null distributes the same 31 flip-count uniformly
over business days in the data range and re-computes precision.

| Variant                                   | Events | Tolerance | Matched | Precision | Null mean | p     |
|-------------------------------------------|-------:|----------:|--------:|----------:|----------:|------:|
| **T1 original** (pre-registered)          |     18 |     +-10d |       5 |    16.1%  |    12.7%  | 0.359 |
| **T1-revised** (post-2022 events only)    |     12 |     +-10d |       5 |    16.1%  |    12.7%  | 0.359 |
| **T1-wide-tolerance** (+-21 calendar d)   |     18 |     +-21d |       6 |    19.4%  |    24.7%  | 0.814 |

All three variants return null. The wide-tolerance variant is
particularly informative: widening the event window makes spurious
matches *easier*, raising the null mean to 24.7%, while observed
precision rises only to 19.4% -- random dates would match the event
list more often than the actual flips do.

The 26 unmatched flips cluster in five windows (2022 Q2-Q3, 2022 Q4 -
2023 Q1, 2023 H2, 2024 H2, 2025 H2 - 2026 Q1) which correspond to
*propagation* phases of headline shocks rather than independent shocks
(see [validation/events_discovered_posthoc.md](../validation/events_discovered_posthoc.md)).
This pattern motivates the duration analysis below.

## 5.6.4 Interpretation: Beyond the Sustained-vs-Transient Split

Three interpretations of the T2 unexpected ordering were entertained:

  (A) **Calibration confound** -- hysteresis tuned on VNINDEX, so the lower
      flip count is an artifact of where the thresholds were set, not a
      real microstructure property. Predicts comparable mean regime
      durations across markets.

  (B) **Monetary stickiness** -- SBV intervenes to suppress regime change;
      VN regimes are "frozen" by policy. Predicts VNINDEX deterministic
      regime mean duration much greater than SPX deterministic mean
      duration; VN stochastic-regime share suppressed.

  (C) **Sustained coordination** -- frontier markets retain coordination
      longer because they lack the institutional arbitrage that breaks
      coordination in real time. Predicts a monotone gradient on T_det:
      VN > SPX > BTC.

To discriminate, we compute mean regime durations from the filtered
label series on each market over the common window:

| Market   | overall_d | T_det_d | T_tra_d | T_sto_d | p(Det) | p(Tra) | p(Sto) | T_det/T_sto |
|----------|----------:|--------:|--------:|--------:|-------:|-------:|-------:|------------:|
| VNINDEX  |    45.53  |  34.10  |  61.74  |  18.83  |  25.7% |  67.8% |   6.5% |        1.81 |
| SP500    |    37.77  |  49.83  |  34.40  |  31.87  |  33.8% |  46.7% |  19.5% |        1.56 |
| BTC      |    30.13  |  48.53  |  19.46  |  30.27  |  46.5% |  32.3% |  21.3% |        1.60 |

(Durations in calendar days; T_det = Deterministic regime mean run
length; T_tra = Transitional; T_sto = Stochastic.)

The data falsify all three interpretations as originally formulated:

  - (A) is rejected: overall mean duration spreads from 30 (BTC) to 46
        (VN), and per-regime durations spread by factors of 2-3x. These
        are not "comparable" in any meaningful sense.
  - (B) is rejected: VN's deterministic regime is the SHORTEST of the
        three (34d vs SPX 50d, BTC 49d). Whatever is anchoring VNINDEX,
        it is not anchoring it in pure-trend states.
  - (C) is rejected: T_det does NOT show a monotone VN > SPX > BTC
        ordering; instead VN is the lowest. The "longer regimes in
        less-arbitraged markets" prediction is wrong on the deterministic
        component.

What the data DO show is a fourth pattern we call
**Transitional Dominance** (colloquially: *Frontier Limbo*):

  - VNINDEX spends 67.8% of its bars in the Transitional regime, with
    Transitional spells averaging 61.7 days -- twice SPX's 34d and
    three times BTC's 19d.
  - VNINDEX almost never enters the Stochastic regime (6.5% of bars vs
    SPX 19.5%, BTC 21.3%) and when it does, exits quickly (18.8d vs
    ~31d on the other two markets).
  - Deterministic spells, when they happen, are SHORTER on VNINDEX
    (34d) than on SPX (50d) or BTC (49d).

Transitional Dominance is consistent with a market that lacks both
decisive directional flow (would produce long Deterministic spells)
and decisive random-walk behaviour (would produce a meaningful
Stochastic share). VNINDEX sits in an indeterminate intermediate
state -- the entropy plane's middle band -- and the low flip rate
observed in T2 reflects long residence in that single regime, not low
overall fragility. The 26 unmatched T1 flips clustering in propagation
windows (Section 5.6.3) is the expected micro-signature of such a
market: each macro shock nudges the system out of and back into the
Transitional band rather than driving a clean Deterministic-to-
Stochastic-and-back excursion.

**Robustness check (T4).** To verify that Transitional Dominance is a
structural property of VNINDEX rather than an artifact of the specific
hysteresis parameterisation the filter was calibrated to, we re-applied
two alternative configurations to the identical fitted GMM:
*Config B* (looser: delta_hard=0.50, delta_soft=0.30, t_persist=6) and
*Config C* (tighter: delta_hard=0.70, delta_soft=0.40, t_persist=10),
alongside the production *Config A* (delta_hard=0.60, delta_soft=0.35,
t_persist=8). The label-share distribution is invariant: p(Det) varies
across {25.7%, 26.2%, 24.9%}, p(Tra) across {67.8%, 66.1%, 68.8%}, and
p(Sto) across {6.5%, 7.7%, 6.4%} -- each within ~1-3 percentage points
of the production values. Filter aggressiveness moves the *timing* of
transitions (annualised flip rate 6.76 -- 11.77 across the three
configs; T_tra 40 -- 72 days) but not the *share* of bars in each
regime. Transitional Dominance is therefore a structural feature of
the VNINDEX entropy plane, not a calibration artifact. Detailed
per-config results in [validation/results/transitional_dominance_robustness.json](../validation/results/transitional_dominance_robustness.json).

## 5.6.5 Synthesis with Paper v1's Entropy Paradox

Paper v1 documented the Entropy Paradox: deterministic-regime return
volatility on VNINDEX is HIGHER than stochastic-regime volatility, the
inverse of the SPX pattern (where the deterministic regime is the
calmer one). Section 5.6.4's Transitional Dominance finding supplies a
microstructural reading of why:

  - VNINDEX's Deterministic share is moderate (25.7%), but Deterministic
    spells are short (34d) and, per paper v1, high-volatility within-
    regime. The market's defining feature is its Transitional dominance
    -- 67.8% of bars with 62-day mean spells, double SPX and triple
    BTC. Deterministic spells, when they happen, look like brief,
    high-amplitude trend bursts (forced liquidations, retail
    capitulations) embedded in this Transitional baseline rather than
    sustained macro-driven trends.

  - SPX enters the Deterministic regime more readily (33.8%) and stays
    longer (50d), with lower within-regime volatility -- consistent with
    macro-driven multi-week trends that institutional flows damp.

  - The Transitional regime, where VNINDEX spends most of its life, is
    not pricing-in indecision so much as sitting in a quasi-stable
    coordination state where price and volume entropies are both in the
    middle band. Liquidations push the system briefly into Det; rare
    flow-shock events push it briefly into Sto; both are exits rather
    than baselines.

The compound-fragility framing follows: VN's risk is NOT "transitions
happen more often" (T2: they don't). It is "the Deterministic regime,
when entered, is high-volatility AND short" -- a thin liquidity layer
across an indeterminate medium, where the deterministic excursions are
the dangerous events.

This refinement does not contradict v1: it specifies that VN's elevated
deterministic volatility (v1) is concentrated in short, infrequent,
liquidation-driven Deterministic spells embedded in a long Transitional
baseline (v7.1). For surveillance, the actionable signal is not "watch
flip rate" but "watch duration of any Deterministic spell -- VN
Deterministic spells longer than ~50 days are out-of-distribution and
warrant escalation."

Transitional Dominance also offers a candidate mechanism for paper v1's
V4 finding (simple-volatility H-statistic outperforms entropy on overall
volatility prediction, while entropy delivers ~5.5x Lift on tail events):
when the Transitional baseline dominates 67.8% of bars, simple volatility
captures the baseline well and entropy's marginal information is
concentrated at *regime identity* (detecting Deterministic entry), not
at regime-average volatility -- exactly where the V4 Lift gap appears.

---

### Cross-references

  - [validation/event_study.py](../validation/event_study.py) (T1 with pre-registered events)
  - [validation/event_study_robustness.py](../validation/event_study_robustness.py) (T1-revised + T1-wide)
  - [validation/cross_market_flip_rate.py](../validation/cross_market_flip_rate.py) (T2)
  - [validation/shuffle_test.py](../validation/shuffle_test.py) (T3)
  - [validation/regime_duration.py](../validation/regime_duration.py) (T-D, this section's discriminator)
  - [validation/transitional_dominance_robustness.py](../validation/transitional_dominance_robustness.py) (T4 robustness check)
  - [validation/events_discovered_posthoc.md](../validation/events_discovered_posthoc.md) (post-hoc cluster log)
  - [validation/results/](../validation/results/) (raw JSON results for all six tests)
