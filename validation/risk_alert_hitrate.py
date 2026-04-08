import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from skills.data_skill import get_latest_market_data
from skills.quant_skill import calc_rolling_wpe, calc_rolling_price_sample_entropy, calc_spe_z
from skills.ds_skill import fit_predict_regime

def run_hitrate_analysis(df: pd.DataFrame) -> None:
    """
    Thực hiện tính toán Max Drawdown tương lai và phân tích Hit Rate theo regime.
    
    Args:
        df: DataFrame chứa dữ liệu giá và các đặc trưng (WPE, SPE_Z) đã loại bỏ NaN.
    """
    # 1. Phân loại Regime sử dụng mô hình GMM trên không gian Entropy
    features = df[["WPE", "SPE_Z"]].values
    _, clf = fit_predict_regime(features, n_components=3)
    
    # Thực hiện dự đoán trên tập dữ liệu và lấy tên regime tương ứng
    labels = clf.predict(features)
    
    # Ánh xạ tên Regime hiện tại sang định dạng được yêu cầu trong logic phân tách (Stable, Fragile, Chaos)
    # Dựa vào mô tả: Deterministic = Chaos, Transitional = Fragile, Stochastic = Stable
    mapping = {
        "Deterministic": "Chaos", 
        "Transitional": "Fragile", 
        "Stochastic": "Stable"
    }
    
    df["RegimeName"] = [mapping.get(clf.get_regime_name(lbl), clf.get_regime_name(lbl)) for lbl in labels]
    
    # 2. Định nghĩa Alerts và tính toán Forward Max Drawdown
    df["alert_chaos"] = df["RegimeName"] == "Chaos"
    df["alert_fragile"] = df["RegimeName"].isin(["Fragile", "Chaos"])
    
    # Max Drawdown trong 5, 10, 20 ngày tương lai
    for window in [5, 10, 20]:
        future_min = df["Close"].shift(-1).rolling(window).min().shift(-(window-1))
        df[f"MaxDrawdown_{window}d"] = (future_min / df["Close"] - 1) * 100

    # Loại bỏ các dòng bị NaN do tính drawdown ở cuối chuỗi dữ liệu
    eval_df = df.dropna(subset=["MaxDrawdown_5d", "MaxDrawdown_10d", "MaxDrawdown_20d"]).copy()
    
    # 3. Phân tích Hit rate theo từng Regime (Dạng Bảng)
    windows = [5, 10, 20]
    thresholds = [(-3, "3%"), (-5, "5%"), (-7, "7%")]
    
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP HIT RATE & LIFT THEO REGIME".center(80))
    print("="*80)
    
    counts = {r: len(eval_df[eval_df["RegimeName"] == r]) for r in ["Stable", "Fragile", "Chaos"]}
    print(f"Tổng số phiên: Stable ({counts['Stable']}), Fragile ({counts['Fragile']}), Chaos ({counts['Chaos']})\n")
    
    print(f"{'Thời gian':<12} | {'Mức sụt giảm':<14} | {'Stable':<10} | {'Fragile':<10} | {'Chaos':<10} | {'Lift (C/S)':<12}")
    print("-" * 80)
    
    for window in windows:
        col = f"MaxDrawdown_{window}d"
        for thresh, label in thresholds:
            rates = {}
            for regime in ["Stable", "Fragile", "Chaos"]:
                subset = eval_df[eval_df["RegimeName"] == regime]
                n = len(subset)
                hits = (subset[col] < thresh).sum()
                rates[regime] = (hits / n * 100) if n > 0 else 0.0
                
            lift = (rates["Chaos"] / rates["Stable"]) if rates["Stable"] > 0 else 0.0
            
            w_str = f"{window} ngày"
            print(f"{w_str:<12} | {'> ' + label:<14} | {rates['Stable']:>6.1f}%   | {rates['Fragile']:>6.1f}%   | {rates['Chaos']:>6.1f}%   | {lift:>6.2f}x")
        print("-" * 80)
    
    # 5. Visualization (Lưu ý: Thư mục lưu hình cần được tạo nếu chưa tồn tại)
    os.makedirs("validation", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(eval_df.index, eval_df["Close"], color="black", linewidth=0.8, alpha=0.7)
    
    chaos_periods = eval_df[eval_df["RegimeName"] == "Chaos"]
    for idx in chaos_periods.index:
        ax.axvline(idx, color="red", alpha=0.03)
        
    big_drops = eval_df[eval_df["MaxDrawdown_10d"] < -3]
    ax.scatter(big_drops.index, big_drops["Close"], color="red", s=8, zorder=5, label=">3% drawdown ahead")
    
    ax.legend(loc="upper right")
    ax.set_title("Regime Classification vs Actual Drawdown Events (-3% in 10 days)")
    ax.set_ylabel("VNIndex")
    
    plt.tight_layout()
    plt.savefig("validation/risk_alert_hitrate.png", dpi=150)
    print("\nVisualization saved to: validation/risk_alert_hitrate.png")
    
    # plt.show() # Tạm thời comment plt.show() trong script tự động chạy


if __name__ == "__main__":
    print("Loading dữ liệu và tính toán chỉ số Entropy...")
    # Lấy dữ liệu VNINDEX từ thời điểm có sẵn tới hiện tại
    try:
        raw_df = get_latest_market_data("VNINDEX", "2015-01-01")
        
        # Tính toán WPE
        log_rets = np.log(raw_df["Close"] / raw_df["Close"].shift(1)).values
        wpe_arr, _ = calc_rolling_wpe(log_rets, m=3, tau=1, window=22)
        raw_df["WPE"] = wpe_arr
        
        # Tính toán SPE_Z
        sampen = calc_rolling_price_sample_entropy(raw_df["Close"].values, window=60)
        raw_df["SPE_Z"] = calc_spe_z(sampen)
        
        valid_data = raw_df.dropna(subset=["WPE", "SPE_Z"]).copy()
        
        if len(valid_data) > 0:
            run_hitrate_analysis(valid_data)
        else:
            print("Không có đủ dữ liệu WPE và SPE_Z để phân tích.")
            
    except Exception as e:
        print(f"Có lỗi xảy ra trong quá trình tính toán: {e}")
