# ARCHITECTURE.md

## Financial Entropy Agent -- Technical Architecture Specification

**Version**: 3.1 (Decoupled Pipeline Refactor)
**Classification**: Dual-Plane Unsupervised Diagnostics + Tri-Vector Composite Risk Engine

---

## Table of Contents

1. [The Great Purge (Deprecation Notice)](#1-the-great-purge)
2. [Conceptual Separation: Planes vs. Vectors](#2-conceptual-separation-planes-vs-vectors)
3. [Module A: Dual-Plane Unsupervised GMM Diagnostics](#3-module-a-dual-plane-unsupervised-gmm-diagnostics)
4. [Module B: Tri-Vector Composite Risk Engine](#4-module-b-tri-vector-composite-risk-engine)
5. [Dynamic Risk Management](#5-dynamic-risk-management)
6. [Agent Orchestrator](#6-agent-orchestrator)
7. [System Pipeline (Visual)](#7-system-pipeline)

---

## 1. The Great Purge

The following legacy Technical Analysis (TA) components have been **permanently deprecated and removed** from the codebase:

| Deprecated Component | Description | Reason for Removal |
|:---|:---|:---|
| `v_streak` / `a_streak` | Velocity and acceleration streak counters | Heuristic; no statistical foundation |
| `5-day Fingerprint` | Sliding window binary feature vectors | Pattern-matching bias; not generalizable |
| `Historical Rhyme Matching` | Cosine/Euclidean similarity search | Data-snooping; overfitting to local history |
| `vstreak_library.pkl` | Local binary cache for pattern storage | Stateful artifact; violates statelessness |
| `IncrementalHybridMemory` | Local disk persistence layer | Replaced by stateless, on-the-fly computation |
| `WPE Sovereignty` | Hardcoded WPE thresholds (0.55, 0.80) | Human-imposed bias; replaced by Tied GMM |
| `KDE Local Minima` | Kernel Density Estimation for WPE boundaries | Still a threshold heuristic; replaced by GMM |
| `Z-score + Sigmoid` | Composite risk normalization pipeline | Assumes normality; replaced by PowerTransformer |

**Design Principle**: The system now operates as a **pure, stateless mathematical engine**. All metrics are computed on-the-fly from raw OHLCV data. No local binary storage, no historical caching, no human-imposed classification boundaries.

---

## 2. Conceptual Separation: Planes vs. Vectors

The system contains two **parallel, independent pipelines** that serve fundamentally different purposes. Conflating them is a critical architectural error.

| Concept | Definition | Count | Purpose |
|:---|:---|:---|:---|
| **Plane** | A 2D scatter plot where an Unsupervised GMM partitions data into visual clusters | **2** (Price, Volume) | Visual structural diagnostics; qualitative regime identification |
| **Vector** | A feature grouping fed into a mathematical formula for continuous risk scoring | **3** (V1, V2, V3) | Quantitative systemic risk measurement; automated 0-100 composite index |

**There is NO "Plane 3".** VN30 Cross-Sectional Breadth (Correlation Entropy, MFI) has no GMM classifier and no visual scatter plot. It exists exclusively as **Vector 3** within the Composite Risk Engine.

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
                  |           |                             +-------------+-------------+
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

---

## 3. Module A: Dual-Plane Unsupervised GMM Diagnostics

Module A provides **visual, qualitative proof** of structural state via two independent 2D phase spaces, each with its own GMM classifier.

### 3.1 Plane 1: Standardized Shock Space (Price Kinematics)

#### 3.1.1 Feature Extraction

**X-Axis: Weighted Permutation Entropy (WPE)**

$$H_{WPE} = -\frac{1}{\ln(m!)} \sum_{w} p_w \cdot \ln(p_w)$$

where $p_w$ are amplitude-weighted permutation probabilities, $m$ is the embedding dimension (default: 3), and the normalization factor $\ln(m!)$ constrains $H \in [0, 1]$.

- $H \to 0$: Perfectly ordered (deterministic trend)
- $H \to 1$: Maximum disorder (stochastic noise)

**Y-Axis: Momentum Entropy Flux**

$$V(t) = \Delta WPE(t) = WPE(t) - WPE(t-1)$$

$$a(t) = \Delta V(t) = V(t) - V(t-1)$$

$$\text{Flux}(t) = \text{sign}(V) \cdot \sqrt{V^2 + a^2} \times 100\%$$

Kinematic interpretation:
- **Flux < 0**: Kinetic energy DECREASING. System cooling, consolidating, stabilizing.
- **Flux > 0**: Kinetic energy INCREASING. System heating, accelerating toward decay.
- **Flux ~ 0**: Kinematic equilibrium. No structural acceleration.

#### 3.1.2 Power Transform (Entropy Shock Normalization)

Financial entropy distributions are inherently **left-skewed and bounded** ($WPE \in [0, 1]$). The system applies `PowerTransformer(method='yeo-johnson', standardize=True)` to both axes **before** GMM fitting.

After transformation, the data represents **"Entropy Shocks"** -- standardized deviations from the expected distributional shape.

#### 3.1.3 Tied-Covariance GMM (Topological Slicing)

```python
GaussianMixture(
    n_components=3,
    covariance_type='tied',   # 3 clusters share 1 covariance matrix
    means_init=[[-1.5, 0], [0, 0], [+1.5, 0]],  # Force left-to-right along X
    max_iter=500,
)
```

**Why `covariance_type='tied'`?**

With `'full'` covariance, each cluster has its own 2x2 matrix. When the Y-axis (Flux) has heavy tails, one cluster can develop a massive Y-variance, creating an ellipse that wraps concentrically around other clusters ("core vs. periphery" topology).

With `'tied'`, all 3 clusters share a **single covariance matrix**. Every cluster has the **same ellipse shape and orientation** -- only the centroid positions differ. Combined with `means_init` spread along the X-axis, the GMM is mathematically forced to partition LEFT-TO-RIGHT.

**Label Assignment Protocol:**
```
1. Sort GMM centroids by X-axis: argsort(means_[:, 0])
2. Map: Lowest X  -> Stable (0)
        Middle X  -> Fragile (1)
        Highest X -> Chaos (2)
```

#### 3.1.4 Confidence Ellipses (95%)

With tied covariance, the ellipse geometry is computed once from the shared matrix $\Sigma$:

```
1. Eigendecomposition: eigenvalues, eigenvectors = eigh(Sigma)
2. Semi-axes: width = 2 * n_std * sqrt(lambda_1), height = 2 * n_std * sqrt(lambda_2)
3. Rotation angle: theta = arctan2(eigenvectors[1,0], eigenvectors[0,0])
```

All 3 ellipses have **identical width, height, and angle** -- placed at different centroid positions.

### 3.2 Plane 2: Volume Entropy Space (Liquidity Structure)

#### 3.2.1 Dual-Path Volume Processing

| Path | Metric | Scale | Purpose |
|:---|:---|:---|:---|
| **Macro** | `Vol_Global_Z` | Global Z-score of log(volume) | Absolute liquidity scale detection |
| **Micro** | `Vol_Rolling_Z` | Rolling Z-score (252-day) | Structural behavior baseline |

#### 3.2.2 Entropy Metrics (computed on Micro path)

- **Shannon Entropy** ($H_{Shannon}$): Histogram-based normalized entropy of rolling volume z-scores. Measures dispersion vs. concentration. $H \in [0, 1]$.

- **Sample Entropy** ($SampEn$): Template-matching complexity of the volume z-score series. Parameters: $m=2$, $r=0.2$. Higher values indicate structural irregularity.

#### 3.2.3 Full-Covariance GMM (Volume Regimes)

Plane 2 uses `GaussianMixture(n_components=3, covariance_type='full')` on `[Vol_Shannon, Vol_SampEn]`. Unlike Plane 1, the volume space does not have the concentric topology problem because both axes have comparable variance scales.

Labels: Consensus Flow, Dispersed Flow, Erratic/Noisy Flow.

#### 3.2.4 Diagnostic Interpretation

| SampEn Level | Global Z | Interpretation |
|:---|:---|:---|
| Low | Negative | Institutional Accumulation (low entropy, below-average volume) |
| Low | Positive | Smart Money Distribution (ordered selling at scale) |
| High | Positive | Climax Distribution (peak FOMO, systemic fragility) |
| High | Negative | Capitulation Chaos (erratic behavior at low volume) |

---

## 4. Module B: Tri-Vector Composite Risk Engine

Module B is a **purely mathematical pipeline** that produces a continuous 0-100 systemic risk score. It has no visual scatter plots and no GMM classifiers. It consumes raw metrics from all three measurement domains (Price, Volume, VN30 Breadth) and synthesizes them into a single composite index.

### 4.1 Vector Definitions

| Vector | Weight | Raw Features | Source Domain |
|:---|:---|:---|:---|
| **V1** (Price Kinematics) | 0.40 | `WPE`, `abs(Momentum_Flux)/100` | Price OHLCV |
| **V2** (Volume Depth) | 0.40 | `Vol_SampEn`, `abs(Vol_Global_Z)`, `Vol_Shannon` | Volume series |
| **V3** (Structural Breadth) | 0.20 | `Corr_Entropy/100`, `MFI` | VN30 cross-section |

**Note:** V3 (VN30 Breadth) has **no visual Plane and no GMM classifier**. It is a raw feature vector fed directly into the composite scoring formula.

### 4.2 Metrics Exclusive to V3

#### 4.2.1 Correlation Entropy (EVD)

$$S_{corr} = -\frac{\sum_i p_i \cdot \ln(p_i)}{\ln(M)} \times 100$$

where $p_i = \frac{\lambda_i}{\sum_j \lambda_j}$ are the normalized eigenvalues of the VN30 Pearson correlation matrix, and $M$ is the number of stocks.

- $S_{corr} < 40$: Heavy-cap consensus (index driven by a few leaders)
- $S_{corr} > 70$: Structural fragmentation (broad-based selling/buying)

#### 4.2.2 Market Fragility Index (MFI)

$$MFI = WPE \times (1 - C_{JS})$$

where $C_{JS}$ is the Jensen-Shannon Statistical Complexity. MFI captures the interaction between disorder ($WPE$) and structural complexity ($C_{JS}$).

### 4.3 Preprocessing Pipeline

```mermaid
graph LR
    A["Raw Features\n7 dimensions"] --> B["PowerTransformer\nyeo-johnson\nper vector"]
    B --> C["MinMaxScaler\n[0, 1]\nper vector"]
    C --> D["Vector Mean\nper vector"]
    D --> E["Weighted Sum\n0.4*V1 + 0.4*V2 + 0.2*V3"]
    E --> F["Composite Score\n0-100"]
    F --> G{"P75 / P90\nRolling 504-day"}
    G -->|"Below P75"| H["STABLE"]
    G -->|"P75 to P90"| I["ELEVATED"]
    G -->|"Above P90"| J["CRITICAL"]
```

### 4.4 Data Leakage Prevention

Each vector's `PowerTransformer` and `MinMaxScaler` are fitted **exclusively** on the rolling 504-day historical window. The current day's features are then **transformed** (not fit-transformed) using these fitted objects.

```python
# Per-vector processing (simplified)
pt = PowerTransformer(method="yeo-johnson", standardize=True)
mms = MinMaxScaler(feature_range=(0, 1))

v_pt = pt.fit_transform(v_hist)       # Fit on 504-day history
v_scaled = mms.fit_transform(v_pt)    # Scale to [0,1]

c_pt = pt.transform(v_current)        # Transform current (NO fit)
c_scaled = mms.transform(c_pt)        # Scale using history params
```

---

## 5. Dynamic Risk Management

### 5.1 Rolling Window

The system uses a **504-day rolling window** (approximately 2 trading years) for all statistical calibrations:

- PowerTransformer fitting per vector
- MinMaxScaler fitting per vector
- Composite score percentile computation

**Rationale**: A 2-year window captures at least one full market cycle while ensuring the model adapts to macroeconomic regime changes and forgets outdated structural patterns.

### 5.2 Percentile Thresholds

| Threshold | Percentile | Risk Label | Interpretation |
|:---|:---|:---|:---|
| Below P75 | < 75th percentile | **STABLE** | Systemic coherence maintained |
| P75 to P90 | 75th - 90th percentile | **ELEVATED** | Structural divergence; monitoring required |
| Above P90 | > 90th percentile | **CRITICAL** | Phase transition zone; top-decile risk event |

These thresholds are **never hardcoded**. They are recomputed for every new data point based on the rolling 504-day composite score history.

### 5.3 Minimum Separation Guard

```python
if critical_bound - elevated_bound < 3.0:
    critical_bound = elevated_bound + 3.0
```

---

## 6. Agent Orchestrator

### 6.1 ReAct Protocol

```
1. fetch_market_data       -> Retrieve OHLCV (VNINDEX) + VN30
2. compute_entropy_metrics -> WPE, MFI, Momentum Entropy Flux
3. compute_volume_entropy  -> Shannon, SampEn, Global Z
4. predict_market_regime   -> Tied GMM Topological Slicing (Plane 1)
5. predict_volume_regime   -> Full GMM Volume Classification (Plane 2)
6. Synthesize              -> Tri-Vector Composite Risk Score (Module B)
```

### 6.2 Reasoning Constraints

- Permitted: *"Entropy Shock"*, *"Phase Transition"*, *"Kinematic Energy"*, *"Systemic Coherence"*, *"Structural Divergence"*, *"GMM Topological Slicing"*
- Forbidden: *"Support level"*, *"Resistance"*, *"Bollinger Bands"*, *"RSI"*, *"Moving Average"*, *"Overbought/Oversold"*, *"thresholds"*, *"cutoffs"*

---

## 7. System Pipeline

```mermaid
graph TD
    subgraph "Data Ingestion"
        A["OHLCV Data\nvnstock / yfinance"] --> B["Log Returns"]
        A --> C["Volume Series"]
        A --> D["VN30 Returns"]
    end

    subgraph "MODULE A: Dual-Plane Unsupervised GMM (Visual Diagnostics)"
        subgraph "Plane 1: Price Kinematics"
            B --> E["Rolling WPE\nm=3, tau=1, w=22"]
            E --> F["Momentum Entropy Flux\nV, a, Flux"]
            F --> H["PowerTransformer\nyeo-johnson"]
            E --> H
            H --> I["Tied GMM n=3\ncovariance_type=tied"]
            I --> J["Regime Labels\nStable / Fragile / Chaos"]
        end
        subgraph "Plane 2: Volume Structure"
            C --> K["log1p Transform"]
            K --> M["Rolling Z-Score\nMicro Structure"]
            M --> N["Shannon Entropy"]
            M --> O["Sample Entropy\nm=2, r=0.2"]
            N --> VOL_GMM["Full GMM n=3\nVolume Regimes"]
            O --> VOL_GMM
        end
    end

    subgraph "MODULE B: Tri-Vector Composite Risk Engine (Mathematical Scoring)"
        E --> S["V1: WPE + Flux"]
        O --> T["V2: SampEn + GZ + Shannon"]
        N --> T
        K --> L["Global Z-Score\nMacro Scale"]
        L --> T
        D --> P["Pearson Correlation\n22-day rolling"]
        P --> Q["Eigenvalue Decomposition"]
        Q --> R["Correlation Entropy\nS_corr 0-100"]
        E --> G["MFI = WPE * (1-C)"]
        R --> U["V3: Corr_Entropy + MFI"]
        G --> U
        S --> V["PowerTransformer + MinMaxScaler\nper vector"]
        T --> V
        U --> V
        V --> W["Weighted Sum\n0.4*V1 + 0.4*V2 + 0.2*V3"]
        W --> X["Composite Score 0-100"]
        X --> Y{"P75 / P90\n504-day Rolling"}
        Y --> Z["STABLE / ELEVATED / CRITICAL"]
    end

    J --> AA["Dashboard\nDual-Plane Scatter Plots\n95% Confidence Ellipses"]
    VOL_GMM --> AA
    Z --> AA
```

---

## Appendix: Dependency Matrix

| Module | File | Dependencies | Purpose |
|:---|:---|:---|:---|
| Data Skill | `skills/data_skill.py` | `vnstock`, `yfinance`, `pandas` | OHLCV retrieval |
| Quant Skill | `skills/quant_skill.py` | `numpy`, `numba`, `pandas` | WPE, SampEn, Shannon, EVD, Flux |
| DS Skill | `skills/ds_skill.py` | `sklearn` (GMM, PowerTransformer), `scipy` | Tied GMM + Volume GMM classification |
| Orchestrator | `agent_orchestrator.py` | `anthropic`, `sklearn` (PowerTransformer, MinMaxScaler) | ReAct + Composite Risk |
| Dashboard | `dashboard.py` | `streamlit`, `plotly` | Dual-Plane visualization terminal |

---

*This document constitutes the complete technical specification of the Financial Entropy Agent v3.1 architecture. "Planes" refer exclusively to 2D visual GMM scatter plots (2 total). "Vectors" refer exclusively to feature groupings for the composite risk formula (3 total). These are parallel, independent pipelines.*
