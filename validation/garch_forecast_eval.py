"""
GARCH Forecast Evaluation -- Financial Entropy Agent
Out-of-sample rolling-window forecast validation: GARCH(1,1) vs 22-day rolling std.

Method:
    1. Load VNINDEX OHLCV from 2020-01-01
    2. Rolling 504-day training window, 1-step-ahead variance forecast
    3. Benchmark: simple 22-day rolling std
    4. QLIKE loss + MSE variance + Pearson correlation
    5. Time-series plot of forecasts vs realized, cumulative QLIKE advantage
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive — avoids blocking on headless terminals
import matplotlib.pyplot as plt
from arch import arch_model

warnings.filterwarnings("ignore")

# --- Allow imports from project root ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.data_skill import get_latest_market_data

TRAIN_WINDOW: int = 504          # 2 trading years
OUTPUT_DIR: str   = os.path.dirname(__file__)


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
def load_data(start_date: str = "2020-01-01") -> pd.DataFrame:
    print(f"[1/6] Fetching VNINDEX from {start_date} ...")
    df = get_latest_market_data(ticker="VNINDEX", start_date=start_date)
    if df is None or len(df) < TRAIN_WINDOW + 50:
        raise RuntimeError(f"Need at least {TRAIN_WINDOW + 50} trading days.")
    df = df.sort_index()
    print(f"      {len(df)} trading days loaded  [{df.index[0].date()} to {df.index[-1].date()}]")
    return df


# ==============================================================================
# STEP 2: ROLLING WINDOW FORECAST
# ==============================================================================
def run_rolling_forecast(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[2/6] Rolling GARCH(1,1) forecast  (window={TRAIN_WINDOW}, step=1) ...")
    log_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100
    n = len(log_rets)

    results: list[dict] = []

    for t in range(TRAIN_WINDOW, n - 1):
        train = log_rets.iloc[t - TRAIN_WINDOW : t]
        actual_return = log_rets.iloc[t]

        try:
            am  = arch_model(train, vol="Garch", p=1, q=1, dist="Normal")
            res = am.fit(disp="off", options={"maxiter": 300})
            forecast = res.forecast(horizon=1)
            sigma_forecast = float(forecast.variance.iloc[-1, 0] ** 0.5)
        except Exception:
            continue

        results.append({
            "date":             log_rets.index[t],
            "actual_return":    float(actual_return),
            "actual_sq_return": float(actual_return ** 2),
            "sigma_forecast":   sigma_forecast,
            "sigma_sq_forecast": sigma_forecast ** 2,
        })

        if len(results) % 100 == 0:
            print(f"  Processed {len(results)} forecasts ...")

    res_df = pd.DataFrame(results).set_index("date")
    print(f"      Total forecasts: {len(res_df)}")
    return res_df, log_rets


# ==============================================================================
# STEP 3: BENCHMARK — 22-DAY ROLLING STD
# ==============================================================================
def add_benchmark(res_df: pd.DataFrame, log_rets: pd.Series) -> pd.DataFrame:
    print("[3/6] Computing 22-day rolling std benchmark ...")
    rolling_std = log_rets.rolling(22).std()
    res_df["benchmark_sigma"] = rolling_std.reindex(res_df.index).values

    # Drop rows where benchmark is NaN (first 22 days of test period)
    res_df = res_df.dropna(subset=["benchmark_sigma"])
    # Guard: replace zero/negative benchmark with tiny floor to avoid log(0)
    res_df["benchmark_sigma"] = res_df["benchmark_sigma"].clip(lower=1e-8)
    print(f"      Benchmark rows after dropna: {len(res_df)}")
    return res_df


# ==============================================================================
# STEP 4: EVALUATION METRICS
# ==============================================================================
def qlike(realized_sq: np.ndarray, forecast_sq: np.ndarray) -> float:
    """
    QLIKE loss: mean(h_t/sigma_t^2 + log(sigma_t^2)) — Patton (2011).
    Equivalent robust form: mean(u - log(u) - 1) where u = realized/forecast.
    Lower is better. Robust to outliers vs MSE.
    """
    forecast_sq = np.clip(forecast_sq, 1e-12, None)   # avoid log(0)
    ratio = realized_sq / forecast_sq
    ratio = np.clip(ratio, 1e-12, None)
    return float(np.mean(ratio - np.log(ratio) - 1))


def evaluate(res_df: pd.DataFrame) -> dict:
    print("[4/6] Computing evaluation metrics ...")

    r2   = res_df["actual_sq_return"].values
    g2   = res_df["sigma_sq_forecast"].values
    b2   = (res_df["benchmark_sigma"].values) ** 2

    qlike_garch = qlike(r2, g2)
    qlike_bench = qlike(r2, b2)

    mse_garch = float(np.mean((r2 - g2) ** 2))
    mse_bench = float(np.mean((r2 - b2) ** 2))

    corr = float(res_df["sigma_forecast"].corr(res_df["actual_return"].abs()))

    print(f"\n  QLIKE - GARCH:     {qlike_garch:.4f}")
    print(f"  QLIKE - Rolling22: {qlike_bench:.4f}")
    print(f"  GARCH better (QLIKE): {qlike_garch < qlike_bench}")

    print(f"\n  MSE Variance - GARCH:     {mse_garch:.4f}")
    print(f"  MSE Variance - Rolling22: {mse_bench:.4f}")

    print(f"\n  Correlation(sigma_forecast, |r_actual|): {corr:.4f}")

    return {
        "qlike_garch": qlike_garch,
        "qlike_bench": qlike_bench,
        "mse_garch":   mse_garch,
        "mse_bench":   mse_bench,
        "corr":        corr,
    }


# ==============================================================================
# STEP 5: VISUALIZATION
# ==============================================================================
def plot_results(res_df: pd.DataFrame) -> None:
    print("[5/6] Generating plots ...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0E1117")
    for ax in axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("#00FF41")

    # --- Top: sigma forecast vs |actual return| ---
    axes[0].plot(res_df.index, res_df["actual_return"].abs(),
                 alpha=0.25, color="#AAAAAA", linewidth=0.8, label="|Actual Return| (%)")
    axes[0].plot(res_df.index, res_df["sigma_forecast"],
                 color="#FF5F1F", linewidth=1.2, label="GARCH(1,1) sigma forecast")
    axes[0].plot(res_df.index, res_df["benchmark_sigma"],
                 color="#00BFFF", linewidth=1.0, alpha=0.6, label="Rolling 22d sigma")
    axes[0].legend(facecolor="#111", labelcolor="white", fontsize=9)
    axes[0].set_ylabel("Volatility (%/day)")
    axes[0].set_title("Out-of-Sample Volatility Forecast: GARCH(1,1) vs Benchmark")
    axes[0].grid(alpha=0.1, color="white")

    # --- Bottom: cumulative QLIKE advantage ---
    g2  = res_df["sigma_sq_forecast"].values.clip(1e-12)
    b2  = (res_df["benchmark_sigma"].values ** 2).clip(1e-12)
    r2  = res_df["actual_sq_return"].values

    def _pointwise_qlike(r2_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
        ratio = np.clip(r2_arr / s2_arr, 1e-12, None)
        return ratio - np.log(ratio) - 1

    ql_garch = _pointwise_qlike(r2, g2)
    ql_bench = _pointwise_qlike(r2, b2)
    cum_diff  = (ql_bench - ql_garch).cumsum()

    axes[1].plot(res_df.index, cum_diff, color="#00FF41", linewidth=1.2)
    axes[1].axhline(0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
    axes[1].fill_between(res_df.index, cum_diff, 0,
                          where=(cum_diff > 0), alpha=0.15, color="#00FF41")
    axes[1].fill_between(res_df.index, cum_diff, 0,
                          where=(cum_diff < 0), alpha=0.15, color="#FF3131")
    axes[1].set_ylabel("Cumulative QLIKE advantage")
    axes[1].set_title("GARCH advantage over benchmark (positive = GARCH better)")
    axes[1].grid(alpha=0.1, color="white")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "garch_forecast_eval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"      Plot saved: {out_path}")


# ==============================================================================
# STEP 6: SUMMARY
# ==============================================================================
def print_summary(res_df: pd.DataFrame, metrics: dict) -> None:
    qg  = metrics["qlike_garch"]
    qb  = metrics["qlike_bench"]
    improvement_pct = (1 - qg / qb) * 100 if qb != 0 else float("nan")

    print("\n" + "=" * 50)
    print("=== GARCH FORECAST EVALUATION ===")
    print("=" * 50)
    print(f"Test period : {res_df.index[0].date()} to {res_df.index[-1].date()}")
    print(f"N forecasts : {len(res_df)}")
    print(f"Train window: {TRAIN_WINDOW} days")
    print()
    print(f"QLIKE  GARCH:     {qg:.4f}")
    print(f"QLIKE  Rolling22: {qb:.4f}")
    print(f"Improvement:      {improvement_pct:+.1f}%  ({'GARCH better' if qg < qb else 'Benchmark better'})")
    print()
    print(f"MSE Variance  GARCH:     {metrics['mse_garch']:.4f}")
    print(f"MSE Variance  Rolling22: {metrics['mse_bench']:.4f}")
    print()
    print(f"Correlation(sigma_forecast, |r_actual|): {metrics['corr']:.4f}")

    if qg < qb and metrics["corr"] > 0.2:
        print("\nPASS  (GARCH outperforms on QLIKE and shows positive forecast correlation)")
    elif qg < qb:
        print("\nPARTIAL PASS  (GARCH better QLIKE, but low forecast correlation)")
    else:
        print("\nFAIL  (Benchmark outperforms GARCH on QLIKE)")


# ==============================================================================
# MAIN
# ==============================================================================
def run_evaluation(start_date: str = "2020-01-01") -> None:
    df = load_data(start_date)
    res_df, log_rets = run_rolling_forecast(df)
    res_df  = add_benchmark(res_df, log_rets)
    metrics = evaluate(res_df)
    plot_results(res_df)
    print_summary(res_df, metrics)


if __name__ == "__main__":
    run_evaluation(start_date="2020-01-01")
