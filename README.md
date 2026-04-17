# Financial Entropy Agent

**Production risk monitoring software using entropy-based regime classification and GARCH-X conditional volatility.**

## Purpose

A deployable risk monitoring system for financial markets combining:
- Information-theoretic regime classification (Weighted Permutation Entropy, Standardized Price Sample Entropy)
- Hysteresis-filtered GMM clustering for stable regime labels
- GARCH-X(1,1) conditional volatility with entropy exogenous features
- Filtered Historical Simulation for tail risk estimation
- Regime x Volatility Verdict Matrix for actionable risk signals

**Target users**: Traders, fund managers, institutional risk teams.

**Companion research**: See [entropy-paradox-research](https://github.com/223hoangthai35/entropy-paradox-research) for academic papers and validation studies.

## Current Status

**v7.1 — Stable production release (frozen)**

Active development paused during developer's MSc program. Production work resumes post-2029.

## Core Features

### Regime Classification
- Raw full-covariance GMM on [WPE, SPE_Z] (Price Plane)
- Yeo-Johnson normalized GMM on [Vol_Shannon, Vol_SampEn] (Volume Plane)
- Hysteresis wrapper: delta_hard=0.60, delta_soft=0.35, t_persist=8

### Conditional Volatility
- GARCH-X(1,1) with entropy exogenous variables
- Automatic p-value pruning (p > 0.10 dropped)
- FHS-based VaR 5% and ES 5%

### Risk Verdict
- 3x4 decision matrix: sigma_adjusted x regime
- Color-coded: LOW to EXTREME RISK

### Dashboard
- Streamlit-based real-time visualization
- Phase space plots, AI narrative generation

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Architecture

See [architecture.md](architecture.md) for detailed system design.

## Reproducibility

Academic paper results reproducible via companion research repo.

## Author

Hoang Thai — Independent development (2026)

## License

MIT License
