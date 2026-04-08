"""
Regime Validation -- Financial Entropy Agent
Tests whether GMM regime labels (Stochastic / Transitional / Deterministic) have real discriminative
power by correlating them against FORWARD realized volatility.

Method:
    1. Load VNINDEX OHLCV from 2020-01-01
    2. Compute WPE + SPE_Z → fit GMM → regime labels
    3. Compute forward realized vol (5d / 10d / 20d)
    4. Group statistics + Kruskal-Wallis non-parametric test
    5. Boxplot + WPE scatter visualization
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

warnings.filterwarnings("ignore")

# --- Allow imports from project root ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.data_skill import get_latest_market_data
from skills.quant_skill import (
    calc_rolling_wpe,
    calc_rolling_price_sample_entropy,
    calc_spe_z,
)
from skills.ds_skill import fit_predict_regime, REGIME_NAMES

# ==============================================================================
# HYPERPARAMETERS (immutable research constants — mirror CLAUDE.md)
# ==============================================================================
WPE_M: int     = 3
WPE_TAU: int   = 1
WPE_WINDOW: int = 22
SPE_WINDOW: int = 60
SPE_M: int     = 2
SPE_R: float   = 0.2

OUTPUT_DIR = os.path.dirname(__file__)


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
def load_data(start_date: str = "2020-01-01") -> pd.DataFrame:
    print(f"[1/7] Fetching VNINDEX from {start_date} …")
    df = get_latest_market_data(ticker="VNINDEX", start_date=start_date)
    if df is None or len(df) < 120:
        raise RuntimeError("Not enough data returned from get_latest_market_data()")
    df = df.sort_index()
    print(f"      {len(df)} trading days loaded  [{df.index[0].date()} to {df.index[-1].date()}]")
    return df


# ==============================================================================
# STEP 2: COMPUTE WPE + SPE_Z → GMM REGIME LABELS
# ==============================================================================
def compute_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/7] Computing WPE …")
    log_rets = np.log(df["Close"] / df["Close"].shift(1)).fillna(0).values
    wpe_arr, _ = calc_rolling_wpe(log_rets, m=WPE_M, tau=WPE_TAU, window=WPE_WINDOW)
    df["WPE"] = wpe_arr

    print("[3/7] Computing SPE_Z …")
    spe_raw = calc_rolling_price_sample_entropy(
        df["Close"].values, window=SPE_WINDOW, m=SPE_M, r_factor=SPE_R
    )
    df["SPE_Z"] = calc_spe_z(spe_raw)

    return df


def fit_regimes(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/7] Fitting GMM regimes (Plane 1: WPE + SPE_Z) …")
    feat_df = df[["WPE", "SPE_Z"]].dropna()
    features = feat_df.values

    labels, _ = fit_predict_regime(features)

    df["RegimeLabel"] = np.nan
    df.loc[feat_df.index, "RegimeLabel"] = labels

    # Semantic names from ds_skill.REGIME_NAMES (canonical source of truth)
    df["RegimeName"] = df["RegimeLabel"].map(REGIME_NAMES)

    counts = df["RegimeName"].value_counts()
    print(f"      Regime counts: {counts.to_dict()}")
    return df


# ==============================================================================
# STEP 3: FORWARD REALIZED VOLATILITY
# ==============================================================================
def compute_forward_vol(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/7] Computing forward realized volatility (5d / 10d / 20d) …")
    log_rets = np.log(df["Close"] / df["Close"].shift(1))

    for window in [5, 10, 20]:
        # std of log returns over the NEXT `window` trading days
        # shift(-1): exclude today; rolling(window): N-day window; shift(-(window-1)): align to today
        df[f"RealVol_{window}d"] = (
            log_rets.shift(-1)
            .rolling(window)
            .std()
            .shift(-(window - 1))
            * np.sqrt(252) * 100
        )

    return df


# ==============================================================================
# STEP 4: GROUP STATISTICS
# ==============================================================================
def compute_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/7] Computing regime statistics …")
    regime_stats = df.groupby("RegimeName").agg({
        "RealVol_5d":  ["mean", "median", "std", "count"],
        "RealVol_10d": ["mean", "median", "std", "count"],
        "RealVol_20d": ["mean", "median", "std", "count"],
    })
    print("\n--- Regime Statistics ---")
    print(regime_stats.to_string())
    return regime_stats


# ==============================================================================
# STEP 5: KRUSKAL-WALLIS TEST
# ==============================================================================
def run_kruskal_wallis(df: pd.DataFrame) -> tuple[float, float]:
    groups = [
        group["RealVol_20d"].dropna().values
        for _, group in df.groupby("RegimeName")
    ]
    # Need at least 2 non-empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        print("[WARN] Not enough regime groups for Kruskal-Wallis test.")
        return float("nan"), float("nan")

    stat, p_value = kruskal(*groups)
    print(f"\nKruskal-Wallis: H={stat:.2f}, p={p_value:.4f}")
    return float(stat), float(p_value)


# ==============================================================================
# STEP 6: VISUALIZATION
# ==============================================================================
def plot_validation(df: pd.DataFrame) -> None:
    print("[7/7] Generating plots …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regime Validation — VNINDEX (2020–present)", fontsize=13, y=1.01)

    # --- Boxplot: RealVol_20d by regime ---
    order = [r for r in ["Stochastic", "Transitional", "Deterministic"] if r in df["RegimeName"].unique()]
    plot_data = [df[df["RegimeName"] == r]["RealVol_20d"].dropna().values for r in order]
    bp = axes[0].boxplot(plot_data, labels=order, patch_artist=True, showfliers=False)

    palette = {"Stochastic": "#2ecc71", "Transitional": "#f39c12", "Deterministic": "#e74c3c"}
    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(palette.get(label, "#999999"))
        patch.set_alpha(0.7)

    axes[0].set_title("Forward 20-day Realized Volatility by Regime")
    axes[0].set_ylabel("Annualized Vol (%)")
    axes[0].set_xlabel("Regime")
    axes[0].grid(axis="y", alpha=0.3)

    # --- Scatter: WPE vs RealVol_20d, coloured by regime ---
    for regime in order:
        subset = df[df["RegimeName"] == regime].dropna(subset=["WPE", "RealVol_20d"])
        axes[1].scatter(
            subset["WPE"], subset["RealVol_20d"],
            c=palette.get(regime, "gray"),
            label=regime, alpha=0.3, s=10
        )
    axes[1].set_xlabel("WPE (Weighted Permutation Entropy)")
    axes[1].set_ylabel("Forward 20d RealVol (%)")
    axes[1].legend(markerscale=2)
    axes[1].set_title("WPE vs Forward Realized Volatility")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "regime_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      Plot saved: {out_path}")
    plt.show()


# ==============================================================================
# STEP 7: SUMMARY PRINT
# ==============================================================================
def print_summary(df: pd.DataFrame, stat: float, p_value: float) -> None:
    print("\n" + "=" * 50)
    print("=== REGIME VALIDATION SUMMARY ===")
    print("=" * 50)
    for regime in ["Stochastic", "Transitional", "Deterministic"]:
        subset = df[df["RegimeName"] == regime]["RealVol_20d"].dropna()
        if len(subset) == 0:
            print(f"{regime}: NO DATA")
            continue
        print(f"{regime:10s}: mean={subset.mean():.2f}%  median={subset.median():.2f}%  n={len(subset)}")

    print(f"\nKruskal-Wallis H={stat:.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("PASS  (p < 0.05 -> regime labels have statistically significant discriminative power)")
    else:
        print("FAIL  (p >= 0.05 -> cannot reject H0 that regime distributions are identical)")


# ==============================================================================
# MAIN
# ==============================================================================
def run_validation(start_date: str = "2020-01-01") -> None:
    df = load_data(start_date)
    df = compute_entropy_features(df)
    df = fit_regimes(df)
    df = compute_forward_vol(df)
    _ = compute_regime_stats(df)
    stat, p_value = run_kruskal_wallis(df)
    plot_validation(df)
    print_summary(df, stat, p_value)


if __name__ == "__main__":
    run_validation(start_date="2020-01-01")
