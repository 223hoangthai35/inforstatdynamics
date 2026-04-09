# Financial Entropy Agent

**An entropy-based market risk surveillance system that uses Information Theory and Statistical Physics to detect structural risk invisible to traditional volatility measures.**

Built by a former securities broker who studied physics — motivated by the observation that financial markets exhibit Type-2 chaos, where dangerous coordination (herding, panic) produces deterministic structure that entropy can detect.

![Regime Validation](validation/regime_validation.png)

---

## The Problem

Traditional risk management relies on volatility — the standard deviation of past returns. This approach has a fundamental blind spot: **volatility is backward-looking**. It tells you the market *was* risky, not that it *is becoming* risky. Before every major crash, rolling volatility is low because the market appears calm.

As a securities broker observing the Vietnamese stock market (VNINDEX), I noticed a pattern: the most dangerous periods weren't when the market was visibly chaotic, but when price movements became unusually *structured* — when everyone was buying the same stocks, following the same momentum, creating an illusion of stability that masked fragility underneath.

This led to a hypothesis rooted in my physics background: **if financial markets are Type-2 chaotic systems, then entropy — the measure of disorder — should detect structural risk before volatility does.**

---

## The Approach

The system measures market disorder through three entropy metrics, classifies structural regimes using unsupervised learning, and estimates conditional volatility via GARCH modeling.

### Entropy Feature Engineering

**Weighted Permutation Entropy (WPE)** measures the ordinal pattern disorder in price log-returns. Low WPE means price movements follow deterministic, repeating patterns — the signature of coordinated behavior (herding, panic, momentum). High WPE means random, unpredictable movements — a normal, healthy market.

**Standardized Price Sample Entropy (SPE_Z)** measures trajectory complexity of close prices. It captures how predictable the price path is in amplitude space, complementing WPE's ordinal analysis.

**Volume Entropy (Shannon + SampEn)** measures liquidity structure — whether capital flow is concentrated (institutional consensus) or dispersed (fragmented, no agreement).

### Regime Classification (GMM)

A Full-Covariance Gaussian Mixture Model (n=3) operates directly on raw [WPE, SPE_Z] features — no preprocessing or normalization — discovering the natural topological boundaries of three market phases:

- **Deterministic** (low entropy) — Strong ordinal structure in price movements. Indicates trending, coordinated behavior. *Highest risk.*
- **Transitional** (mid entropy) — Phase boundary between ordered and disordered states. *Moderate risk.*
- **Stochastic** (high entropy) — Random walk behavior. Normal, healthy market. *Lowest risk.*

### Unsupervised Regime Discovery — Visual Evidence

The GMM discovers natural cluster boundaries in raw entropy feature space without any human-imposed thresholds or preprocessing.

**Price Entropy Phase Space** (Plane 1: WPE × SPE_Z)

![Price Phase Space](docs/images/price_phase_space.png)

The scatter plot shows three distinct clusters emerging from raw entropy features:
- **Red (Deterministic)**: Low WPE, negative SPE_Z — structured, predictable price patterns indicating coordinated behavior
- **Yellow (Transitional)**: Mid-range entropy — the boundary between ordered and disordered states
- **Green (Stochastic)**: High WPE, positive SPE_Z — random, complex price evolution characteristic of normal market conditions

The dashed ellipses represent 95% confidence boundaries of each GMM component's full covariance matrix — each cluster has its own shape and orientation, capturing the true geometry of entropy distributions.

**Volume Entropy Phase Space** (Plane 2: Shannon × SampEn)

![Volume Phase Space](docs/images/volume_phase_space.png)

Volume flow structure reveals three liquidity regimes:
- **Blue (Consensus Flow)**: Low entropy — capital moves in organized, predictable patterns (institutional consensus)
- **Purple (Dispersed Flow)**: Mid entropy — fragmented capital flow, no clear institutional agreement
- **Red (Erratic/Noisy Flow)**: High entropy — chaotic liquidity with unpredictable volume impulses

**Market Structure Timeline**

![Market Structure](docs/images/market_structure.png)

The regime classification overlaid on VNINDEX price history (2020–2026) demonstrates how entropy regimes align with market events. Note that Deterministic (red) periods coincide with both sharp rallies and sharp declines — entropy measures structural coordination, not direction.

### Conditional Volatility (GARCH)

GARCH(1,1) provides the conditional volatility backbone, with entropy features tested as exogenous variables (GARCH-X). Filtered Historical Simulation computes Expected Shortfall (ES 5%) without assuming Gaussian tails.

### AI Explanation Layer

An LLM agent (Claude API) translates quantitative signals into natural-language risk narratives for non-technical investors — the "last mile" between mathematical models and actionable advice.

For the full mathematical formulations (WPE, SampEn, Yeo-Johnson, GMM specifications, GARCH variance equations), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Validation Results

All validation is out-of-sample on VNINDEX data (2020–2026, ~1,600 trading days). Code is in the `validation/` folder and fully reproducible.

### V1: Regime Labels Discriminate Future Volatility

Entropy-based regime labels significantly predict forward 20-day realized volatility.

| Regime | Mean Forward 20d Vol | Median | n |
|:-------|:---------------------|:-------|:--|
| **Deterministic** (High Risk) | 21.59% | 20.23% | 464 |
| Transitional | 18.00% | 16.11% | 754 |
| **Stochastic** (Low Risk) | 15.15% | 11.54% | 340 |

**Kruskal-Wallis H = 150.89, p < 0.0001** — Regime labels carry statistically significant information about future market risk.

> **Note:** "Deterministic" does not mean "market will fall." It means "market structure is highly coordinated" — which can manifest as aggressive rallies *or* sharp declines. The higher forward volatility (21.59%) reflects the fact that coordinated states produce larger moves in *both* directions, consistent with the Type-2 chaos framework where structural order precedes instability.

### V3: Tail Risk Detection — Where Entropy Excels

The system's discriminative power *increases with event severity* — precisely the behavior required for tail risk management.

| Timeframe | Drawdown | Stochastic | Deterministic | Lift |
|:----------|:---------|:-----------|:--------------|:-----|
| 5 days | > 3% | 8.5% | 17.4% | **2.06×** |
| 5 days | > 7% | 0.8% | 4.3% | **5.50×** |
| 10 days | > 5% | 6.3% | 16.0% | **2.54×** |
| 20 days | > 7% | 6.5% | 19.6% | **3.00×** |

When the system identifies a Deterministic regime, the probability of a severe drawdown (>7% in 5 days) is **5.5 times higher** than during Stochastic regimes. For a risk manager, this is actionable information for position sizing.

![Risk Alert Hit Rate](validation/risk_alert_hitrate.png)

### V4: Entropy vs Simple Volatility — Complementary, Not Replacement

| Model | Features | H-statistic | p-value |
|:------|:---------|:------------|:--------|
| A: Entropy | WPE, SPE_Z | 150.55 | < 0.0001 |
| B: Simple Vol | Rolling 22d vol, 5d vol change | 299.78 | < 0.0001 |
| C: Combined | WPE, SPE_Z, Rolling 22d vol | 229.04 | < 0.0001 |

Simple volatility discriminates future volatility better (H=299.78 vs 150.55) — this is expected because past volatility auto-correlates with future volatility. **However, entropy measures a different dimension of risk: structural fragility.** Entropy detects FOMO peaks, herding behavior, and momentum concentration *before* they manifest as volatility spikes. The Combined model (H=229.04) confirms entropy carries complementary information beyond what simple volatility provides.

![Entropy vs Simple Vol](validation/entropy_vs_simple.png)

### V2: GARCH Forecast Evaluation — Honest Limitations

| Metric | GARCH(1,1) | Rolling 22d | Winner |
|:-------|:-----------|:------------|:-------|
| QLIKE | 1.7765 | 1.5737 | Rolling 22d |
| MSE Variance | 15.5130 | 15.3941 | Rolling 22d |
| Correlation(σ, \|r\|) | 0.3302 | — | — |

GARCH(1,1) underperforms simple rolling volatility on point forecast metrics for VNINDEX. This is consistent with frontier market characteristics — VNINDEX exhibits jump risk (circuit breakers, policy shocks) that GARCH's smooth conditional variance cannot capture. The positive directional correlation (r=0.33) confirms the model tracks volatility direction correctly but misestimates magnitude.

This result justifies the system's use of Filtered Historical Simulation rather than parametric VaR — FHS does not assume Gaussian tails and is more robust to the fat-tailed distribution of VNINDEX returns.

![GARCH Forecast Evaluation](validation/garch_forecast_eval.png)

---

## Key Insight: The Entropy Paradox

The most important finding from validation was counterintuitive: **in financial markets, low entropy = high risk, and high entropy = low risk.**

This is the *opposite* of physical systems, where low entropy indicates calm equilibrium. In financial markets — which are Type-2 chaotic systems with reflexive, adversarial participants — low entropy means price movements have become deterministic. Deterministic structure in financial prices is not a sign of health; it is the signature of **coordinated behavior**: herding, panic selling, or momentum-driven rallies. These states are inherently unstable and produce the largest realized moves.

Maximum entropy, by contrast, means maximum randomness — a market where diverse participants with diverse strategies cancel each other out. This is the *healthy* state.

**Entropy does not measure volatility. It measures the absence of dangerous coordination.**

This insight directly connects to the Type-2 chaos hypothesis that motivated the project: in a system where participants observe and react to each other (reflexivity), order is a warning signal.

**Practical implication:** Because Deterministic regimes coincide with both sharp rallies *and* sharp declines, the system is not a directional predictor. It is a **structural fragility detector**. When the market enters a Deterministic regime during a rally, it does not mean "sell now" — it means "the current trend is driven by coordination rather than diverse conviction, making it vulnerable to sharp reversals." This distinction between directional prediction and structural assessment is critical for responsible deployment in financial applications.

---

## Lessons Learned

**1. Validation changes everything.** The initial regime labels were inverted (mislabeled "Stable" for the highest-risk regime). Without forward-looking validation, this error would have gone undetected and the system would have given exactly wrong advice. Lesson: never trust model outputs without out-of-sample evidence.

**2. Complexity must justify itself.** GARCH-X with entropy exogenous variables was statistically insignificant during calm periods — entropy adds no value to volatility forecasting when the market is already stable. The system needed to be redesigned with adaptive activation: entropy contributes through regime classification (always active) rather than forcing it into the variance equation.

**3. Simple benchmarks are essential.** Rolling 22-day volatility outperformed GARCH on VNINDEX. This doesn't invalidate the entropy approach — it clarifies its role: entropy excels at structural risk detection (Lift 5.5× for tail events), while simple volatility excels at point forecasting. They solve different problems.

**4. Domain knowledge matters more than model sophistication.** The Entropy Paradox was only interpretable because of my background in physics (understanding entropy in different system types) and finance (understanding that coordinated behavior drives crashes). A purely technical approach would have either missed the inversion or abandoned entropy as "broken."

---

## Mathematical Foundation

### Weighted Permutation Entropy (WPE)

For a time series `{x_t}`, compute ordinal patterns of length `m` (embedding dimension) and weight each pattern by its amplitude variance:

```
WPE = - sum_pi [ w(pi) * log(w(pi)) ]

where:
  pi     = ordinal pattern (permutation of m consecutive values)
  w(pi)  = sum of variances of windows matching pattern pi
         / total variance of all windows
```

**Parameters**: m=3, tau=1 (lag), window=22 days (1 trading month). The amplitude weighting distinguishes WPE from standard PE — large-amplitude patterns (panic moves, rallies) carry more weight than small random fluctuations.

**Interpretation**: WPE in [0, log(m!)] normalized to [0, 1]. Low WPE = repeating ordinal patterns = coordinated, deterministic behavior. High WPE = diverse patterns = normal random market.

### Sample Entropy (SampEn)

SampEn measures the probability that two similar m-length templates remain similar at length m+1:

```
SampEn(m, r) = -log( A / B )

where:
  B = number of template pairs within tolerance r at length m
  A = number of template pairs within tolerance r at length m+1
  r = 0.2 * std(window)   (adaptive tolerance, 20% of local sigma)
```

**Parameters**: m=2, r=0.2*sigma, window=60 days. SPE_Z is the rolling Z-score of SampEn: `(SampEn_t - mean_t) / std_t`.

**Interpretation**: Low SampEn = highly predictable price trajectory (low complexity). High SampEn = complex, irregular path (high complexity). Unlike WPE (ordinal structure), SampEn captures amplitude-space complexity.

### Shannon Volume Entropy

```
H_Shannon = - sum_i [ p_i * log(p_i) ]

where p_i = probability mass in bin i of the normalized volume distribution
```

Applied to rolling 60-day volume windows. Measures whether capital flow is concentrated (few dominant volume days → low entropy) or dispersed (uniform distribution → high entropy).

### GMM Regime Classifier

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| n_components | 3 | Three regimes from physics analogy: ordered / transition / disordered |
| covariance_type | full | Each cluster has its own shape and orientation — no isotropy assumption |
| n_init | 10 | Multiple initializations to escape local optima |
| Preprocessing | None (Plane 1) | Raw [WPE, SPE_Z] features — the natural scale carries physical meaning |
| Features | [WPE, SPE_Z] | Two orthogonal entropy axes: ordinal disorder × trajectory complexity |

The full covariance matrix allows GMM to discover the true geometric structure of entropy distributions — elongated, rotated clusters that axis-aligned covariance would miss.

### GARCH(1,1) Variance Equation

```
sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

where:
  omega  = long-run variance floor
  alpha  = reaction coefficient (sensitivity to shocks)
  beta   = persistence coefficient (variance memory)
  alpha + beta < 1  (stationarity constraint)
```

**Filtered Historical Simulation**: Rather than assuming Gaussian tails, standardized residuals `z_t = epsilon_t / sigma_t` are drawn from empirical historical distribution. VaR 5% and ES 5% are computed from quantiles of `sigma_t * z_{historical}`. This handles VNINDEX's fat tails and jump risk (circuit breakers, policy shocks) that parametric VaR cannot capture.

---

## Architecture Overview

The system processes a single OHLCV stream through six sequential pipeline layers:

```
LAYER 1 — DATA INGESTION
  skills/data_skill.py
  vnstock (VNINDEX, VN30 constituents) + yfinance fallback
  Output: OHLCV DataFrame, VN30 returns matrix

LAYER 2 — ENTROPY FEATURE ENGINE
  skills/quant_skill.py
  WPE (m=3, tau=1, window=22)         — ordinal disorder in price returns
  SPE_Z (m=2, r=0.2*sigma, window=60) — trajectory complexity, standardized
  Vol_Shannon + Vol_SampEn (window=60) — liquidity structure entropy
  Cross-sectional entropy (VN30)       — breadth / coordination signal
  MFI                                  — money flow intensity
  Output: entropy feature columns on df

LAYER 3 — UNSUPERVISED REGIME CLASSIFIER
  skills/ds_skill.py
  Plane 1: Full-Cov GMM(n=3) on raw [WPE, SPE_Z]
    -> Deterministic / Transitional / Stochastic
  Plane 2: Full-Cov GMM(n=3) on [Vol_Shannon, Vol_SampEn]
    -> Consensus Flow / Dispersed Flow / Erratic/Noisy Flow
  No preprocessing, no normalization — discovers natural topology
  Output: RegimeName, VolRegimeName columns on df

LAYER 4 — CONDITIONAL VOLATILITY ENGINE
  agent_orchestrator.py  fit_garch_x()
  GARCH(1,1) fitted on full log-return history
  Filtered Historical Simulation -> VaR 5%, ES 5%
  Regime multiplier applied: sigma_adj = sigma_raw x multiplier
    Stochastic: 1.0x  |  Transitional: 1.4x  |  Deterministic: 2.2x
  Output: sigma_daily_pct, sigma_annual_pct, ES_5pct, ES_5pct_adjusted

LAYER 5 — VERDICT MATRIX (Regime-Aware Risk Classification)
  dashboard.py  (inline logic, post-GARCH)
  Combines sigma_adjusted level with price regime in a 3×4 matrix:
    Stochastic + any sigma  -> ELEVATED VOLATILITY (yellow) at most
    Transitional + elevated -> HIGH RISK (orange)
    Deterministic + low     -> STRUCTURAL WARNING (orange) — calm-before-storm
    Deterministic + high    -> EXTREME RISK (red)
  Prevents sigma threshold alone from mislabeling liquidity spikes as systemic crises.
  Fallback when GARCH unavailable: calc_composite_risk_score() entropy aggregate (0-100)
  Output: risk_verdict, risk_verdict_color, mult_explain

LAYER 6 — AI EXPLANATION LAYER
  agent_orchestrator.py  ReAct loop (Claude API)
  5-tool sequence: fetch -> entropy -> volume -> regime -> vol_regime
  Synthesizes all layers into structured markdown risk narrative
  Direction-aware: Deterministic + rally vs Deterministic + decline
  Output: natural-language report for non-technical stakeholders
```

```
skills/data_skill.py  ->  skills/quant_skill.py  ->  skills/ds_skill.py
     LAYER 1                    LAYER 2                   LAYER 3
                                     |
                           agent_orchestrator.py + dashboard.py
                           LAYER 4 + 5 + 6 (ReAct + Verdict Matrix)
                                     |
                               dashboard.py
                           Streamlit + Plotly UI
```

For detailed mathematical specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Structure

```
Financial Entropy Agent/
├── agent_orchestrator.py       # GARCH engine, risk scoring, AI agent orchestrator
├── dashboard.py                # Streamlit interactive terminal
├── skills/
│   ├── data_skill.py           # Data ingestion (vnstock, yfinance)
│   ├── quant_skill.py          # WPE, SampEn, Shannon, kinematics
│   └── ds_skill.py             # GMM regime classification
├── validation/
│   ├── regime_validation.py    # V1: Regime labels vs forward realized vol
│   ├── garch_forecast_eval.py  # V2: GARCH out-of-sample forecast
│   ├── risk_alert_hitrate.py   # V3: Drawdown prediction hit rate
│   └── entropy_vs_simple.py    # V4: Entropy vs simple vol comparison
├── ARCHITECTURE.md             # Full mathematical specifications
├── requirements.txt
└── README.md                   # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- An Anthropic API key (for the AI agent; optional — system works without it)

### Installation

```bash
git clone https://github.com/223hoangthai35/financial-entropy-agent.git
cd financial-entropy-agent
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard.py
```

### Run Validation Suite

```bash
python validation/regime_validation.py
python validation/risk_alert_hitrate.py
python validation/entropy_vs_simple.py
python validation/garch_forecast_eval.py
```

---

## Technical Requirements

`numpy`, `pandas`, `numba` (JIT), `scikit-learn` (GMM), `scipy`, `arch` (GARCH), `statsmodels`, `plotly`, `streamlit`, `anthropic` (optional), `matplotlib`, `vnstock`, `yfinance`

---

## Generalization to Other Markets

The system is designed as a market-agnostic entropy surveillance engine. VNINDEX is the validation market, not an architectural constraint.

### What Is Market-Specific

| Component | What changes | What stays the same |
|:----------|:-------------|:--------------------|
| Data source | `data_skill.py` — replace vnstock with Bloomberg/Reuters/yfinance | OHLCV schema |
| Regime multipliers | 1.0 / 1.4 / 2.2 calibrated on VNINDEX tail behavior | GARCH model specification |
| GARCH fit | Refits on new market's return distribution automatically | Model specification |
| VN30 breadth | Replace with S&P 500 constituents, FTSE 100, etc. | Eigenvalue decomposition logic |

### Verifying the Entropy Paradox on a New Market

The most important validation before deploying on a new market is to re-confirm that **low entropy = high risk** holds in that market's regime structure. This is not guaranteed — markets with different microstructure (e.g., thin frontier markets with high manipulation risk vs. deep liquid markets) may exhibit different entropy-risk relationships.

**Recommended steps:**

1. Run `validation/regime_validation.py` on the new market's data
2. Check that the Kruskal-Wallis H-statistic is significant (p < 0.05)
3. Check that the **ordering** of mean forward volatility matches: Deterministic > Transitional > Stochastic
4. If ordering is inverted, the GMM cluster-to-regime mapping (`_cluster_to_regime` in `ds_skill.py`) needs to be re-calibrated for that market's entropy topology

### Markets Where This Approach Is Well-Motivated

The entropy framework is most applicable to markets exhibiting **Type-2 chaos characteristics**:
- Markets with identifiable herding and momentum-following behavior
- Markets where retail participation is high (coordinated behavior easier to detect)
- Markets with sufficient history for GMM convergence (minimum ~500 trading days recommended)

For deep, institutionally-dominated markets (e.g., US Treasuries), the Entropy Paradox may be weaker because arbitrage suppresses coordination faster. For frontier markets with thin liquidity, jump risk dominates and GARCH limitations (noted in V2) become more severe.

---

## References

- Bandt, C. & Pompe, B. (2002). *Permutation Entropy: A Natural Complexity Measure for Time Series.* Physical Review Letters, 88(17).
- Fadlallah, B. et al. (2013). *Weighted-Permutation Entropy: A Complexity Measure for Time Series Incorporating Amplitude Information.* Physical Review E, 87(2).
- Richman, J.S. & Moorman, J.R. (2000). *Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy.* American Journal of Physiology.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics.

---

## About the Author

Physics background -> Securities broker -> Data Science.

This project was born from observing that traditional technical analysis systematically failed to warn about structural market risks. The hypothesis — that entropy from statistical physics could detect dangerous coordination in financial markets — was validated through rigorous out-of-sample testing on 6 years of VNINDEX data.

---

*Disclaimer: This system is a quantitative research tool for structural risk assessment. It is not investment advice. All investment decisions require final approval from qualified professionals.*
