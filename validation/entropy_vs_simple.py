import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.data_skill import get_latest_market_data
from skills.quant_skill import (
    calc_rolling_wpe, calc_rolling_price_sample_entropy, calc_spe_z
)
from skills.ds_skill import fit_predict_regime, REGIME_NAMES

# ==============================================================================
# 1. LOAD DATA & COMPUTE FEATURES
# ==============================================================================
print("Loading VNINDEX data...")
df = get_latest_market_data(ticker="VNINDEX", start_date="2020-01-01")
print(f"Loaded {len(df)} rows: {df.index[0]} to {df.index[-1]}")

log_rets = np.log(df["Close"] / df["Close"].shift(1))

# --- Entropy features (Model A) ---
wpe_arr, c_arr = calc_rolling_wpe(log_rets.values, m=3, tau=1, window=22)
sampen_price = calc_rolling_price_sample_entropy(df["Close"].values, window=60)
spe_z = calc_spe_z(sampen_price)
df["WPE"] = wpe_arr
df["SPE_Z"] = spe_z

# --- Simple volatility features (Model B) ---
df["RollingVol22"] = log_rets.rolling(22).std() * np.sqrt(252) * 100  # annualized %
df["VolChange5"] = df["RollingVol22"].pct_change(5)  # 5-day momentum of vol

# --- Forward realized volatility (ground truth) ---
df["RealVol_20d"] = log_rets.shift(-1).rolling(20).std().shift(-19) * np.sqrt(252) * 100

# ==============================================================================
# 2. FIT 3 GMM MODELS
# ==============================================================================

# Model A: Entropy-based (WPE, SPE_Z) — current system
print("\n--- Model A: Entropy [WPE, SPE_Z] ---")
valid_A = df.dropna(subset=["WPE", "SPE_Z", "RealVol_20d"]).copy()
features_A = valid_A[["WPE", "SPE_Z"]].values
labels_A, clf_A = fit_predict_regime(features_A, n_components=3)
valid_A["regime"] = [clf_A.get_regime_name(l) for l in labels_A]
print(f"  Samples: {len(valid_A)}")
print(f"  Regime distribution: {valid_A['regime'].value_counts().to_dict()}")

# Model B: Simple volatility (RollingVol22, VolChange5)
print("\n--- Model B: Simple Vol [RollingVol22, VolChange5] ---")
valid_B = df.dropna(subset=["RollingVol22", "VolChange5", "RealVol_20d"]).copy()
features_B = valid_B[["RollingVol22", "VolChange5"]].values
labels_B, clf_B = fit_predict_regime(features_B, n_components=3)
valid_B["regime"] = [clf_B.get_regime_name(l) for l in labels_B]
print(f"  Samples: {len(valid_B)}")
print(f"  Regime distribution: {valid_B['regime'].value_counts().to_dict()}")

# Model C: Combined (WPE, SPE_Z, RollingVol22)
print("\n--- Model C: Combined [WPE, SPE_Z, RollingVol22] ---")
valid_C = df.dropna(subset=["WPE", "SPE_Z", "RollingVol22", "RealVol_20d"]).copy()
features_C = valid_C[["WPE", "SPE_Z", "RollingVol22"]].values
labels_C, clf_C = fit_predict_regime(features_C, n_components=3)
valid_C["regime"] = [clf_C.get_regime_name(l) for l in labels_C]
print(f"  Samples: {len(valid_C)}")
print(f"  Regime distribution: {valid_C['regime'].value_counts().to_dict()}")

# ==============================================================================
# 3. DISCRIMINATIVE POWER — Kruskal-Wallis H-statistic
# ==============================================================================
print("\n" + "=" * 60)
print("DISCRIMINATIVE POWER: Kruskal-Wallis H-statistic")
print("(Higher H = regime labels discriminate forward vol better)")
print("=" * 60)

results = {}

for name, valid_df in [("A: Entropy", valid_A), 
                         ("B: Simple Vol", valid_B), 
                         ("C: Combined", valid_C)]:
    groups = [g["RealVol_20d"].dropna().values 
              for _, g in valid_df.groupby("regime") if len(g) > 10]
    
    if len(groups) >= 2:
        H, p = kruskal(*groups)
    else:
        H, p = 0, 1.0
    
    results[name] = {"H": H, "p": p, "n": len(valid_df)}
    
    # Per-regime stats
    print(f"\n{name}: H={H:.2f}, p={p:.6f}")
    for regime, group in valid_df.groupby("regime"):
        vol = group["RealVol_20d"].dropna()
        print(f"  {regime:20s}: mean={vol.mean():.2f}%, median={vol.median():.2f}%, n={len(vol)}")

# ==============================================================================
# 4. REGIME SEPARATION — Mean difference between riskiest and safest
# ==============================================================================
print("\n" + "=" * 60)
print("REGIME SEPARATION: Spread between highest-risk and lowest-risk regime")
print("=" * 60)

for name, valid_df in [("A: Entropy", valid_A), 
                         ("B: Simple Vol", valid_B), 
                         ("C: Combined", valid_C)]:
    means = valid_df.groupby("regime")["RealVol_20d"].mean()
    spread = means.max() - means.min()
    print(f"  {name}: max={means.max():.2f}%, min={means.min():.2f}%, spread={spread:.2f}pp")

# ==============================================================================
# 5. EARLY DETECTION — Regime change BEFORE volatility spike
# ==============================================================================
print("\n" + "=" * 60)
print("EARLY DETECTION: Does regime change lead or lag volatility?")
print("=" * 60)

# Tính cross-correlation giữa regime label (numeric) và forward realized vol
# Nếu regime change DẪN TRƯỚC vol spike → entropy có early warning value

for name, valid_df in [("A: Entropy", valid_A), ("B: Simple Vol", valid_B)]:
    # Encode regime as risk score: Deterministic=2, Transitional=1, Stochastic=0
    risk_map = {"Deterministic": 2, "Transitional": 1, "Stochastic": 0}
    risk_score = valid_df["regime"].map(risk_map).fillna(1)
    
    # Correlation at different lags
    print(f"\n{name}:")
    for lag in [0, 5, 10, 20]:
        if lag == 0:
            corr = risk_score.corr(valid_df["RealVol_20d"])
            print(f"  Lag {lag:2d}d (concurrent):  corr = {corr:+.4f}")
        else:
            # Regime today → vol N days later
            shifted_vol = valid_df["RealVol_20d"].shift(-lag)
            corr = risk_score.corr(shifted_vol)
            print(f"  Lag {lag:2d}d (regime leads): corr = {corr:+.4f}")

# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Entropy vs Simple Vol vs Combined — Regime Discrimination of Forward 20d Volatility",
             fontsize=13, fontweight="bold")

colors = {"Deterministic": "#FF3131", "Transitional": "#FFD700", "Stochastic": "#00FF41"}

for idx, (name, valid_df) in enumerate([("A: Entropy [WPE, SPE_Z]", valid_A), 
                                          ("B: Simple [Vol22, VolΔ5]", valid_B),
                                          ("C: Combined", valid_C)]):
    ax = axes[idx]
    
    regime_order = ["Deterministic", "Transitional", "Stochastic"]
    data_groups = []
    labels_plot = []
    color_list = []
    
    for regime in regime_order:
        subset = valid_df[valid_df["regime"] == regime]["RealVol_20d"].dropna()
        if len(subset) > 0:
            data_groups.append(subset.values)
            labels_plot.append(f"{regime}\n(n={len(subset)})")
            color_list.append(colors.get(regime, "gray"))
    
    bp = ax.boxplot(data_groups, labels=labels_plot, patch_artist=True, 
                     widths=0.6, showfliers=False)
    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    H_val = results[list(results.keys())[idx]]["H"]
    p_val = results[list(results.keys())[idx]]["p"]
    ax.set_title(f"{name}\nH={H_val:.1f}, p={'<0.0001' if p_val < 0.0001 else f'{p_val:.4f}'}")
    ax.set_ylabel("Forward 20d Realized Vol (%)" if idx == 0 else "")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("validation/entropy_vs_simple.png", dpi=150, bbox_inches="tight")
print(f"\nChart saved: validation/entropy_vs_simple.png")
# plt.show() # Tạm thời comment plt.show() trong script tự động chạy

# ==============================================================================
# 7. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON SUMMARY")
print("=" * 60)

print(f"\n{'Model':<20s} {'H-statistic':>12s} {'p-value':>12s} {'Winner':>10s}")
print("-" * 56)

best_H = 0
best_name = ""
for name, r in results.items():
    winner = ""
    if r["H"] > best_H:
        best_H = r["H"]
        best_name = name
    print(f"{name:<20s} {r['H']:>12.2f} {r['p']:>12.6f}")

print(f"\n>>> BEST MODEL: {best_name} (H={best_H:.2f})")

# Entropy advantage
H_A = results["A: Entropy"]["H"]
H_B = results["B: Simple Vol"]["H"]
if H_A > H_B:
    pct = (H_A / H_B - 1) * 100
    print(f">>> Entropy advantage over Simple Vol: +{pct:.1f}% better discrimination")
elif H_B > H_A:
    pct = (H_B / H_A - 1) * 100
    print(f">>> Simple Vol advantage over Entropy: +{pct:.1f}% better discrimination")
else:
    print(f">>> Equal discrimination power")

H_C = results["C: Combined"]["H"]
if H_C > max(H_A, H_B):
    print(f">>> Combined model is BEST — entropy + vol together outperform both alone")
    print(f">>> This suggests entropy provides COMPLEMENTARY information to volatility")
