# Financial Entropy Agent

### A Tri-Vector Kinematic Surveillance System for Systemic Risk Quantification

---

## Abstract

**Financial Entropy Agent** is an institutional-grade systemic risk surveillance engine that decodes market microstructure through the principles of **Symbolic Dynamics**, **Information Theory**, and **Non-linear Physics**. The system replaces all heuristic-based Technical Analysis (TA) with continuous, mathematically rigorous models rooted in statistical mechanics.

The core innovation is the **Standardized Entropy Shock Space** -- a power-transformed phase space where a tied-covariance Gaussian Mixture Model (GMM) performs topological slicing along the WPE Shock axis, partitioning the market into structural regimes (Stable, Fragile, Chaos) without any human-imposed boundary constraints.

---

## Core Philosophy: The Paradigm Shift

Traditional financial analysis relies on pattern-matching heuristics: moving averages, RSI thresholds, support/resistance levels. These tools assume stationarity and normality in price series -- assumptions that are systematically violated by real markets.

This system operates on a fundamentally different premise:

| Dimension | Traditional TA | Financial Entropy Agent |
|:---|:---|:---|
| **Signal** | Price level, amplitude | Ordinal pattern entropy (WPE) |
| **Dynamics** | Trend-following | Kinematic velocity and acceleration of entropy |
| **Regime Detection** | Fixed thresholds | Unsupervised GMM in power-transformed space |
| **Risk Scoring** | Rule-based if/else | Continuous composite index (0-100) |
| **Normalization** | Z-score (assumes normality) | Yeo-Johnson PowerTransform (handles skew) |
| **Threshold Calibration** | Expert guesses | Empirical quantiles (P75/P90) from rolling distribution |

---

## Key Features

### 1. Standardized Entropy Shock Space
Weighted Permutation Entropy (WPE) and its kinematic derivatives are projected into a Gaussian-normalized phase space via `PowerTransformer(yeo-johnson)`. A tied-covariance `GaussianMixture(n_components=3, covariance_type='tied')` performs topological slicing along the WPE Shock axis, partitioning the space into three structural regimes via centroid sorting -- no hardcoded thresholds, no human bias. The `tied` constraint forces all three clusters to share a single covariance matrix, preventing the EM algorithm from creating concentric (core-vs-periphery) topologies.

### 2. Tri-Vector Composite Risk Index
Systemic risk is quantified through three orthogonal measurement vectors:
- **V1 -- Price Kinematics (40%)**: WPE magnitude + Momentum Entropy Flux direction.
- **V2 -- Liquidity Depth (40%)**: Sample Entropy + Global Z-Score + Shannon Entropy of volume.
- **V3 -- Structural Breadth (20%)**: Cross-Sectional Correlation Entropy (VN30 EVD) + Market Fragility Index.

Each vector is independently power-transformed and min-max scaled to `[0, 1]` before weighted summation.

### 3. Dynamic Rolling Probability Thresholds
Risk boundaries are not static. The system computes a rolling 504-day (2 trading years) distribution of composite scores and derives:
- **P75**: Elevated risk threshold (top 25% historical scores).
- **P90**: Critical risk threshold (top 10% extreme events).

This ensures the model adapts autonomously to evolving market regimes.

### 4. Autonomous AI Orchestrator
An Anthropic-powered ReAct agent autonomously sequences data retrieval, entropy computation, regime classification, and composite risk synthesis. The agent's reasoning is constrained to statistical physics terminology -- no TA jargon permitted.

---

## Architecture Overview

```
                        FINANCIAL ENTROPY AGENT
                    =======================================

                    [ RAW OHLCV & VN30 DATA ]
                               |
                               +-------------------------------------------------+
                               |                                                 |
                  [ MODULE A: UNSUPERVISED GMM ]                [ MODULE B: COMPOSITE RISK ENGINE ]
                  (Dual-Plane Visual Diagnostics)               (Tri-Vector Mathematical Synthesis)
                               |                                                 |
                         +-----+-----+                             +-------------+-------------+
                         |           |                             |             |             |
                      PLANE 1     PLANE 2                      VECTOR 1      VECTOR 2      VECTOR 3
                      (Price)     (Volume)                     (Price)       (Volume)     (VN30 Breadth)
                         |           |                             |             |             |
                     WPE Shock    Shannon                     WPE / Flux    SampEn / GZ    CorrEnt / MFI
                     Flux Shock   SampEn                          |             |             |
                         |           |                            +-------------+-------------+
                     Tied GMM     Full GMM                                      |
                    (3 Regimes)  (3 Regimes)                         PowerTransformer (Yeo-Johnson)
                                                                                |
                                                                         MinMaxScaler [0, 1]
                                                                                |
                                                              Weighted Sum: 0.4*V1 + 0.4*V2 + 0.2*V3
                                                                                |
                                                                 Composite Risk Score (0-100)
                                                                                |
                                                                 P75/P90 Rolling 504-day
                                                                                |
                                                              STABLE / ELEVATED / CRITICAL
```

For the full technical specification, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Technical Requirements

- **Python**: 3.9+
- **Core Dependencies**: `numpy`, `pandas`, `numba` (JIT optimization), `scikit-learn` (GMM, PowerTransformer, MinMaxScaler)
- **Dashboard**: `streamlit`, `plotly`
- **AI Orchestrator**: `anthropic` (requires `ANTHROPIC_API_KEY`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Launch the Dark Quant Terminal
streamlit run dashboard.py
```

---

## Project Structure

```
Financial Entropy Agent/
|-- agent_orchestrator.py    # Tri-Vector Composite Risk Engine + ReAct Orchestrator
|-- dashboard.py             # Dark Quant Terminal (Streamlit + Plotly)
|-- ARCHITECTURE.md          # Detailed technical architecture document
|-- README.md                # This file
|-- skills/
|   |-- data_skill.py        # Data ingestion (vnstock, yfinance)
|   |-- quant_skill.py       # WPE, Momentum Flux, Volume Entropy, EVD
|   |-- ds_skill.py          # PowerTransform + Tied GMM Classifiers
```

---

## Institutional Use Cases & Practical Applications

This system is not a signal generator. It is a **macro-structural risk filter** designed to operate upstream of any trading strategy, providing a continuous, physics-based assessment of systemic market health.

### 1. Systemic Risk Filter for Dynamic Position Sizing

Quantitative funds and systematic portfolio managers can integrate the **Composite Risk Score (0-100)** into their position sizing and margin allocation frameworks. Rather than relying on subjective judgment calls, the score provides a mathematically derived, rolling measure of aggregate systemic stress.

| Composite Risk Score | Recommended Action |
|:---|:---|
| **0 - 40 (Low)** | Full allocation (100%). Market structure exhibits systemic coherence. Entropy is low, liquidity is orderly. |
| **40 - P75 (Moderate)** | Standard allocation. Monitor Momentum Flux direction for trajectory changes. |
| **P75 - P90 (Elevated)** | Reduce gross exposure by 50%. Halt new position entries. Structural divergence detected across one or more planes. |
| **> P90 (Critical)** | Maximum risk reduction. Initiate hedging protocols. Phase transition imminent -- top-decile extreme event in the rolling 504-day distribution. |

The dynamic P75/P90 boundaries ensure the system self-calibrates to the prevailing volatility regime. During prolonged low-volatility periods, the thresholds compress, making the engine more sensitive to nascent structural shifts.

### 2. Detecting Liquidity Divergences (Trap Detection)

The decoupled multi-plane architecture enables the detection of **hollow rallies (bull traps)** and **false breakdowns** that single-indicator systems miss:

- **Hollow Rally**: Plane 1 (Price) shows "Stable" regime, but Plane 2 (Volume) simultaneously classifies as "Erratic/Dispersed". The price stability is not supported by consensus liquidity flow -- volume entropy is fractured, indicating that the rally is structurally unsupported and vulnerable to sudden reversal.

- **Capitulation Vacuum**: Plane 1 shows "Chaos" but `Vol_Global_Z` is negative (below-average absolute volume). High price entropy is driven by illiquidity and thin order books, not by genuine institutional selling. This distinction is critical for identifying bear-market bottoming processes.

- **Climax Distribution**: High Composite Risk score but `Vol_Global_Z` is strongly positive. Excess liquidity is flowing into a structurally deteriorating market -- characteristic of peak FOMO before a distribution top. The system explicitly flags this as a bubble-peak topology, not a crash signal.

### 3. Measuring Internal Structural Health (Sector Rotation Detection)

Vector 3 (VN30 Breadth) provides a unique diagnostic layer invisible to index-level analysis:

- **Correlation Entropy < 40%**: The VN30's movement is dominated by a narrow set of heavyweight pillars (e.g., VIC, VHM, VCB). The index appears stable, but internal breadth is collapsed. This "propped-up" structure is inherently fragile -- if the leading heavyweights falter, the index has no broad-based support to absorb the shock.

- **Correlation Entropy > 70%**: The component stocks are moving in highly independent directions. This signals aggressive internal sector rotation or capital flight among heavy-cap constituents. The main index may appear range-bound while aggressive structural recomposition occurs beneath the surface.

By monitoring this metric alongside Planes 1 and 2, a portfolio manager receives advance warning of internal fracturing **before** it manifests as visible index-level volatility.

---

## Academic References

- Bandt, C. & Pompe, B. (2002). *Permutation Entropy: A Natural Complexity Measure for Time Series*. Physical Review Letters, 88(17).
- Fadlallah, B. et al. (2013). *Weighted-Permutation Entropy: A Complexity Measure for Time Series Incorporating Amplitude Information*. Physical Review E, 87(2).
- Richman, J.S. & Moorman, J.R. (2000). *Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy*. American Journal of Physiology.
- Yeo, I.K. & Johnson, R.A. (2000). *A New Family of Power Transformations to Improve Normality or Symmetry*. Biometrika, 87(4).

---

**Disclaimer**: This system is designed as a quantitative research tool for institutional systemic risk surveillance. It is not a trading signal generator. All investment decisions based on its output require final approval from qualified human analysts.
