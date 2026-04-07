"""
Financial Entropy Agent -- Tri-Vector Composite Risk Terminal
Dual Pipeline (API/Upload), Kinematic GMM Scatter, Composite Risk Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Import backend skills
from skills.data_skill import get_latest_market_data, fetch_vn30_returns
from skills.quant_skill import (
    calc_rolling_wpe, calc_mfi, calc_correlation_entropy,
    calc_rolling_volume_entropy, calc_momentum_entropy_flux,
)
from skills.ds_skill import fit_predict_regime, fit_predict_volume_regime
from agent_orchestrator import calc_composite_risk_score

# ==============================================================================
# UI CONFIGURATION & MULTILINGUAL SUPPORT
# ==============================================================================
st.set_page_config(page_title="Financial Entropy Agent | Terminal", layout="wide", page_icon="⚡")

# Custom Styling (Dark Quant Terminal)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime&family=Inter:wght@400;800&display=swap');
    
    .reportview-container { background: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFFFFF; font-family: 'Courier Prime', monospace; }
    .metric-label { font-size: 0.9rem; color: #AAAAAA; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 400; }
    
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier Prime', monospace; text-transform: uppercase; letter-spacing: 2px; }
    
    .agent-log {
        background-color: #0a0a0a;
        border-left: 4px solid #00FF41;
        padding: 30px;
        font-family: 'Courier Prime', monospace;
        color: #d1ffd1;
        font-size: 0.95rem;
        line-height: 1.7;
        border-radius: 4px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        border: 1px solid #1a3a1a;
    }
    .agent-log h3 { color: #39FF14 !important; margin-bottom: 20px; border-bottom: 1px solid #1a3a1a; padding-bottom: 10px; }
    .agent-log code { color: #FFD700 !important; font-weight: 800; background: transparent; padding: 0; }
    .agent-log strong { color: #39FF14 !important; text-transform: uppercase; }
    .agent-log table { border-collapse: collapse; width: 100%; margin: 20px 0; border: 1px solid #1a3a1a; }
    .agent-log th, .agent-log td { border: 1px solid #1a3a1a; padding: 12px; text-align: left; }
    .agent-log th { background: #112211; color: #00FF41; }
    
    .arch-badge {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        transition: all 0.3s ease;
    }
    .arch-badge:hover { border-color: #00FF41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.2); }
    
    [data-testid="column"]:nth-child(1) > div:nth-child(1) {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 5px;
    }
    
    .stPlotlyChart { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# UI INITIALIZATION & MULTILINGUAL SUPPORT
# ==============================================================================
risk_score = 0.0
synthesis_label = "STABLE"
risk_color = "#00FF41"
current_wpe = 0.5
current_regime = "STABLE"
current_vol_global_z = 0.0
current_vol_shannon = 0.5
current_vol_regime_name = "CONSENSUS FLOW"
vol_sh_kpi = "0.50"
vol_se_kpi = "0.50"
vol_gz_kpi = "0.00 Z"

def T(en: str, vn: str) -> str:
    return en if st.session_state.get("lang", "EN") == "EN" else vn

if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

# ==============================================================================
# DATA PIPELINE (CACHED)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_and_compute_data(start_date_str, end_date_str, file_bytes=None, file_name=None):
    df = None
    
    # DUAL PIPELINE: Uploaded CSV vs API
    if file_bytes is not None:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
            
        date_candidates = [c for c in df.columns if str(c).lower().strip() in ("date", "time", "ngay", "ngày")]
        if date_candidates:
            df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]])
            df.set_index(date_candidates[0], inplace=True)
            df.sort_index(inplace=True)
            
        col_map = {}
        for c in df.columns:
            key = str(c).lower().strip()
            if key == "open": col_map[c] = "Open"
            elif key == "high": col_map[c] = "High"
            elif key == "low": col_map[c] = "Low"
            elif key == "close": col_map[c] = "Close"
            elif key == "volume": col_map[c] = "Volume"
        df.rename(columns=col_map, inplace=True)
        df.ffill(inplace=True)
    else:
        df = get_latest_market_data(ticker="VNINDEX", start_date=start_date_str, end_date=end_date_str)
        
    df = df.loc[start_date_str:end_date_str].copy()
    if df.empty:
        return df

    df["SMA20"] = df["Close"].rolling(20).mean()
    
    # 1. Compute WPE, Complexity, MFI
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).values
    wpe_arr, c_arr = calc_rolling_wpe(log_returns, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)
    
    df["WPE"] = wpe_arr
    df["Complexity"] = c_arr
    df["MFI"] = mfi_arr
    
    # 2. Kinematic Momentum Entropy Flux
    vel, acc, flux = calc_momentum_entropy_flux(wpe_arr)
    df["PE_Velocity"] = vel
    df["PE_Acceleration"] = acc
    df["Momentum_Entropy_Flux"] = flux
    
    # 3. Volume Entropy Plane (Macro-Micro Fusion)
    if "Volume" in df.columns:
        vol_shannon, vol_sampen, vol_global_z, vol_rolling_z = calc_rolling_volume_entropy(
            df["Volume"].values, window=60, z_window=252
        )
        df["Vol_Shannon"] = vol_shannon
        df["Vol_SampEn"] = vol_sampen
        df["Vol_Global_Z"] = vol_global_z
        df["Vol_Rolling_Z"] = vol_rolling_z
    else:
        df["Vol_Shannon"] = np.nan
        df["Vol_SampEn"] = np.nan
        df["Vol_Global_Z"] = np.nan
        df["Vol_Rolling_Z"] = np.nan
    
    # 4. VN30 Cross-Sectional Entropy
    try:
        vn30_rets = fetch_vn30_returns(start_date=start_date_str, end_date=end_date_str)
        cross_entropy = calc_correlation_entropy(vn30_rets, window=22)
        df["Cross_Sectional_Entropy"] = cross_entropy.reindex(df.index).ffill()
    except Exception as e:
        df["Cross_Sectional_Entropy"] = np.nan
        
    # 5. Predict Price Regime (Tied GMM in Standardized Shock Space)
    price_clf = None
    valid_df = df.dropna(subset=["WPE", "Momentum_Entropy_Flux"]).copy()
    if not valid_df.empty:
        features = valid_df[["WPE", "Momentum_Entropy_Flux"]].values
        labels, price_clf = fit_predict_regime(features, n_components=3)
        valid_df["RegimeLabel"] = labels
        valid_df["RegimeName"] = [price_clf.get_regime_name(lbl) for lbl in labels]
        # Luu transformed data cho scatter plot
        X_tf = price_clf.transform(features)
        valid_df["WPE_Transformed"] = X_tf[:, 0]
        valid_df["Flux_Transformed"] = X_tf[:, 1]
    
    # Luu classifier vao df.attrs de truy cap cho ellipse rendering
    df.attrs["price_classifier"] = price_clf
    
    df["RegimeName"] = np.nan
    df["RegimeLabel"] = np.nan
    if not valid_df.empty:
        df.loc[valid_df.index, "RegimeName"] = valid_df["RegimeName"]
        df.loc[valid_df.index, "RegimeLabel"] = valid_df["RegimeLabel"]
        if "WPE_Transformed" in valid_df.columns:
            df.loc[valid_df.index, "WPE_Transformed"] = valid_df["WPE_Transformed"]
            df.loc[valid_df.index, "Flux_Transformed"] = valid_df["Flux_Transformed"]
    df["RegimeName"] = df["RegimeName"].ffill()
    df["RegimeLabel"] = df["RegimeLabel"].ffill()
    
    # 6. Volume Regime (GMM -- Plane 2)
    vol_valid = df.dropna(subset=["Vol_Shannon", "Vol_SampEn"]).copy()
    if not vol_valid.empty and len(vol_valid) >= 10:
        vol_features = vol_valid[["Vol_Shannon", "Vol_SampEn"]].values
        vol_labels, vol_clf = fit_predict_volume_regime(vol_features, n_components=3)
        vol_valid["VolRegimeLabel"] = vol_labels
        vol_valid["VolRegimeName"] = [vol_clf.get_regime_name(lbl) for lbl in vol_labels]
    
    df["VolRegimeName"] = np.nan
    df["VolRegimeLabel"] = np.nan
    if not vol_valid.empty and len(vol_valid) >= 10:
        df.loc[vol_valid.index, "VolRegimeName"] = vol_valid["VolRegimeName"]
        df.loc[vol_valid.index, "VolRegimeLabel"] = vol_valid["VolRegimeLabel"]
    df["VolRegimeName"] = df["VolRegimeName"].ffill()
    df["VolRegimeLabel"] = df["VolRegimeLabel"].ffill()
    
    return df

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("### 🌐 " + T("Language / Ngôn ngữ", "Language / Ngôn ngữ"))
lang = st.sidebar.radio("", ["EN", "VN"], index=0 if st.session_state["lang"] == "EN" else 1, horizontal=True, label_visibility="collapsed")
st.session_state["lang"] = lang

st.sidebar.markdown(f"### ⚙️ {T('SYSTEM CONFIGURATION', 'CẤU HÌNH HỆ THỐNG')}")

start_date = st.sidebar.date_input(T("Start Date", "Ngày Bắt Đầu"), datetime(2020, 1, 1))
end_date = st.sidebar.date_input(T("End Date", "Ngày Kết Thúc"), datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('1. DUAL PIPELINE: API OR UPLOAD', '1. DUAL PIPELINE: API HOẶC TẢI LÊN')}**")
uploaded_file = st.sidebar.file_uploader(T("Upload custom OHLCV (.csv/.xlsx)", "Tải lên Dữ liệu OHLCV (.csv/.xlsx)"), type=["csv", "xlsx"])

file_bytes = None
file_name = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title(T("FINANCIAL ENTROPY AGENT: SYSTEM ARCHITECT", "FINANCIAL ENTROPY AGENT: SYSTEM ARCHITECT"))
st.markdown(T(
    "Tri-Vector Composite Risk Surveillance: Price Entropy, Liquidity Structure, VN30 Breadth.",
    "Giam sat Rui ro Hop thanh Tri-Vector: Entropy Gia, Cau truc Thanh khoan, Do rong VN30."
))

with st.spinner(T("Computing Tri-Vector Composite Engine...", "Đang tính toán Tri-Vector Composite Engine...")):
    df = load_and_compute_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), file_bytes, file_name)

if df is None or df.empty:
    st.error(T("No data available for the selected dates.", "Không có dữ liệu cho khoảng thời gian này."))
    st.stop()
    
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==============================================================================
# QUANTITATIVE VECTOR EXTRACTION
# ==============================================================================
current_wpe = latest.get("WPE", 0.5)
current_mfi = latest.get("MFI", 0.5)
current_cse = latest.get("Cross_Sectional_Entropy", 50.0)
current_flux = latest.get("Momentum_Entropy_Flux", 0.0)
current_pe_v = latest.get("PE_Velocity", 0.0)
current_pe_a = latest.get("PE_Acceleration", 0.0)

# Plane 2
current_vol_shannon = latest.get("Vol_Shannon", float("nan"))
current_vol_sampen = latest.get("Vol_SampEn", float("nan"))
current_vol_global_z = latest.get("Vol_Global_Z", float("nan"))

# Regime Labels
current_regime = str(latest.get("RegimeName", "Calculating...")).replace("nan", "Calculating...")
current_vol_regime_name = str(latest.get("VolRegimeName", "Calculating...")).replace("nan", "Calculating...")

# KPI Strings
vol_sh_kpi = f"{current_vol_shannon:.2f}" if pd.notna(current_vol_shannon) else "N/A"
vol_se_kpi = f"{current_vol_sampen:.2f}" if pd.notna(current_vol_sampen) else "N/A"
vol_gz_kpi = f"{current_vol_global_z:+.2f}" if pd.notna(current_vol_global_z) else "N/A"

# Run Composite Risk Score
risk_score, synthesis_label, vector_info = calc_composite_risk_score(latest.to_dict(), df=df)
contributions = vector_info.get("contributions", {})
dominant_vector = vector_info.get("dominant_vector", "N/A")

# Dynamic Thresholds
dyn_thresholds = vector_info.get("thresholds", {})
elevated_bound = dyn_thresholds.get("elevated_bound", 55.0)
critical_bound = dyn_thresholds.get("critical_bound", 70.0)
price_clf = df.attrs.get("price_classifier", None)

# Risk Color
if "CRITICAL" in synthesis_label:
    risk_color = "#FF0000"
elif "ELEVATED" in synthesis_label:
    risk_color = "#FF5F1F"
else:
    risk_color = "#00FF41"

# ==============================================================================
# TOP KPI SECTION: THE GRID OF 4
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

# --- Column 1: Systemic Risk Gauge ---
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': risk_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, elevated_bound], 'color': 'rgba(0, 255, 65, 0.1)'},
                {'range': [elevated_bound, critical_bound], 'color': 'rgba(255, 215, 0, 0.1)'},
                {'range': [critical_bound, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
        }
    ))
    
    fig_gauge.update_layout(
        autosize=True,
        height=150, 
        margin=dict(l=20, r=20, t=20, b=20), 
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"{risk_score:.1f}",
            x=0.5, y=0.35,
            xref="paper", yref="paper",
            font=dict(size=45, color="#FFFFFF", family="Courier Prime"),
            showarrow=False,
            xanchor="center",
            yanchor="middle"
        )]
    )
    
    st.markdown(f'<div class="metric-label" style="text-align:center; padding-top:10px;">{T("COMPOSITE RISK SCORE", "DIEM RUI RO HOP THANH")}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"<div style='text-align:center; color:{risk_color}; font-weight:800; font-size:1.0rem; margin-top:-20px;'>{synthesis_label}</div>", unsafe_allow_html=True)

# --- Column 2: Dominant Risk Vector ---
with col2:
    dominant_display = dominant_vector.replace("V1_Price", "V1: PRICE").replace("V2_Volume", "V2: VOLUME").replace("V3_Breadth", "V3: BREADTH")
    dominant_val = contributions.get(dominant_vector, 0)
    contrib_pct = f"{dominant_val * 100:.1f}%"
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("DOMINANT RISK VECTOR", "VECTOR RUI RO CHU DAO")}</div>
        <div class="metric-value" style="color: {risk_color}; font-size: 1.2rem; padding: 5px;">{dominant_display}</div>
        <div style="font-size: 1.0rem; color: #FFD700; margin-top: 5px;">Scaled: {contrib_pct}</div>
        <div style="font-size: 0.75rem; color: #888; margin-top: 5px;">V1={contributions.get('V1_Price',0):.2f} | V2={contributions.get('V2_Volume',0):.2f} | V3={contributions.get('V3_Breadth',0):.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Column 3: Price Dynamics (Plane 1) ---
with col3:
    p1_color = "#00FF41" if "STABLE" in current_regime.upper() else ("#FFD700" if "FRAGILE" in current_regime.upper() else "#FF3131")
    flux_display = f"{current_flux:+.2f}%" if pd.notna(current_flux) else "N/A"
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("PRICE DYNAMICS", "DONG LUC GIA")}</div>
        <div class="metric-value" style="color: {p1_color}; font-size: 1.4rem;">{current_regime}</div>
        <div style="font-size: 0.9rem; color: #888; margin-top: 5px;">WPE: {current_wpe:.4f}</div>
        <div style="font-size: 0.9rem; color: #888;">Flux: {flux_display}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Column 4: Liquidity Depth ---
with col4:
    gz_val = current_vol_global_z if pd.notna(current_vol_global_z) else 0.0
    gz_color = "#00FF41" if abs(gz_val) < 1.0 else ("#FFD700" if abs(gz_val) < 2.0 else "#FF3131")
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("LIQUIDITY DEPTH", "DO SAU THANH KHOAN")}</div>
        <div class="metric-value" style="color: {gz_color}; font-size: 1.6rem;">{vol_gz_kpi}</div>
        <div style="font-size: 0.9rem; color: #888; margin-top: 10px;">Shannon: {vol_sh_kpi}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# VISUALS: Market Structure
# ==============================================================================
st.markdown("---")
st.subheader(T("1. MARKET STRUCTURE", "1. CẤU TRÚC THỊ TRƯỜNG"))

fig1 = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
)

# --- Row 1 (Price) ---
fig1.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="VNindex", increasing_line_color='#FFFFFF', decreasing_line_color='#888888'
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=df.index, y=df['SMA20'], mode='lines', name='SMA20',
    line=dict(color='yellow', width=1, dash='dash')
), row=1, col=1)

# --- Row 2 (WPE) ---
fig1.add_trace(go.Scatter(
    x=df.index, y=df['WPE'], mode='lines', name='WPE (Entropy)',
    line=dict(color='#FF5F1F', width=2)
), row=2, col=1)

# Regime Background Shading
regime_colors = {
    "Stable": "rgba(0, 255, 65, 0.15)",
    "Fragile": "rgba(255, 215, 0, 0.15)",
    "Chaos": "rgba(255, 0, 0, 0.15)",
    "Calculating...": "rgba(128, 128, 128, 0)"
}

# Dummy legend traces
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(0, 255, 65, 1)'), name='Stable'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 215, 0, 1)'), name='Fragile'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 0, 0, 1)'), name='Chaos'))

# Regime shading on Row 1
df['Regime_Shift'] = df['RegimeName'] != df['RegimeName'].shift(1)
shift_indices = df.index[df['Regime_Shift']].tolist()

for i in range(len(shift_indices)):
    start = shift_indices[i]
    end = shift_indices[i+1] if i+1 < len(shift_indices) else df.index[-1]
    regime = df.loc[start, 'RegimeName']
    color = regime_colors.get(regime, "rgba(0,0,0,0)")
    
    fig1.add_vrect(
        x0=start, x1=end, fillcolor=color, opacity=1.0, layer="below", line_width=0,
        row=1, col=1
    )

fig1.update_layout(
    title=dict(text="VNindex Structure State (Tied GMM Regime)", x=0.5, y=0.98, xanchor="center", yanchor="top"),
    template="plotly_dark", height=600, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1.0),
    margin=dict(l=20, r=20, b=20, t=60)
)
fig1.update_xaxes(rangeslider_visible=False)
fig1.update_yaxes(title_text="VNIndex Price", row=1, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
fig1.update_yaxes(title_text="WPE Entropy", row=2, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
st.plotly_chart(fig1, use_container_width=True)

# --- VN30 Cross-sectional Chart ---
st.markdown("---")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df['Cross_Sectional_Entropy'], mode='lines', name='VN30 Cross-sectional Entropy',
    line=dict(color='#00FFFF', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)'
))

fig2.add_trace(go.Scatter(
    x=df.index, y=df['MFI'] * 100, mode='lines', name='MFI',
    line=dict(color='#FFD700', width=1, dash='dot')
))

fig2.update_layout(
    title=dict(text="Cross-sectional Entropy VN30 (Eigenvalue Decomposition)", x=0.5, y=0.95, xanchor="center", yanchor="top"),
    template="plotly_dark", height=400, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=20, r=20, b=20, t=60),
    yaxis=dict(title_text="Entropy (0-100 Scale)")
)
fig2.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# UNSUPERVISED LEARNING PROOF -- DUAL-PLANE
# ==============================================================================
st.markdown("---")
st.subheader(T("2. UNSUPERVISED LEARNING: DUAL-PLANE DS PROOF", "2. BANG CHUNG HOC MAY: HAI MAT PHANG ENTROPY"))
st.markdown(T(
    "Plane 1: Tied GMM Topological Slicing (X=WPE Shock, Y=Flux Shock). Plane 2: Volume GMM (X=Shannon, Y=SampEn).",
    "Mat phang 1: Tied GMM Topological Slicing (X=WPE Shock, Y=Flux Shock). Mat phang 2: Volume GMM (X=Shannon, Y=SampEn)."
))

col_price_plot, col_vol_plot = st.columns([1, 1])

# --- PLOT 1: Price Dynamics Plane (Kinematic) ---
with col_price_plot:
    st.markdown(f"**{T('PLANE 1: STANDARDIZED SHOCK SPACE', 'MẶT PHẲNG 1: KHONG GIAN SHOCK CHUAN HOA')}**")
    plot_df = df.dropna(subset=['WPE_Transformed', 'Flux_Transformed', 'RegimeName'])
    if not plot_df.empty:
        color_map_price = {
            "Stable": "#00FF41",
            "Fragile": "#FFD700",
            "Chaos": "#FF0000",
        }
        scatter_price = px.scatter(
            plot_df, x="WPE_Transformed", y="Flux_Transformed",
            color="RegimeName",
            color_discrete_map=color_map_price,
            hover_data=["Close", "WPE", "Momentum_Entropy_Flux", "MFI"],
            labels={"WPE_Transformed": "Standardized WPE Shock", "Flux_Transformed": "Standardized Flux Shock"},
        )
        # Y=0 horizontal equilibrium line
        scatter_price.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", annotation_text="Equilibrium")

        # 95% Confidence Ellipses (Tied GMM: shared shape, different centers)
        if price_clf is not None:
            def get_tied_ellipse_params(gmm, n_std=2.0):
                """Tinh ellipse params cho Tied GMM (1 covariance chung, 3 centroids)."""
                cov = gmm.covariances_  # Shape: (2, 2) -- ma tran duy nhat
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * n_std * np.sqrt(eigenvalues)
                ellipses = []
                for i in range(gmm.n_components):
                    mean_x, mean_y = gmm.means_[i]
                    ellipses.append({
                        'x0': mean_x, 'y0': mean_y,
                        'width': width, 'height': height, 'angle': theta,
                    })
                return ellipses

            try:
                tied_ellipses = get_tied_ellipse_params(price_clf.gmm, n_std=2.0)
                regime_colors = {0: "#00FF41", 1: "#FFD700", 2: "#FF0000"}
                t_arr = np.linspace(0, 2 * np.pi, 100)
                for cluster_idx, ell in enumerate(tied_ellipses):
                    regime_idx = price_clf._cluster_to_regime.get(cluster_idx, cluster_idx)
                    cos_a = np.cos(np.radians(ell["angle"]))
                    sin_a = np.sin(np.radians(ell["angle"]))
                    x_ell = (ell["width"] / 2) * np.cos(t_arr)
                    y_ell = (ell["height"] / 2) * np.sin(t_arr)
                    x_rot = cos_a * x_ell - sin_a * y_ell + ell["x0"]
                    y_rot = sin_a * x_ell + cos_a * y_ell + ell["y0"]
                    scatter_price.add_trace(go.Scatter(
                        x=x_rot, y=y_rot, mode='lines',
                        line=dict(color=regime_colors.get(regime_idx, "white"), width=1.5, dash='dash'),
                        showlegend=False, hoverinfo='skip',
                    ))
            except Exception:
                pass

        scatter_price.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="Price Regime (Tied GMM)",
            margin=dict(l=20, r=20, b=20, t=20), height=450,
        )
        st.plotly_chart(scatter_price, use_container_width=True)

# --- PLOT 2: Volume Entropy Plane ---
with col_vol_plot:
    st.markdown(f"**{T('PLANE 2: LIQUIDITY STRUCTURE', 'MẶT PHẲNG 2: CẤU TRÚC THANH KHOẢN')}**")
    vol_plot_df = df.dropna(subset=['Vol_Shannon', 'Vol_SampEn', 'VolRegimeName'])
    if not vol_plot_df.empty:
        color_map_vol = {
            "Consensus Flow": "#1E90FF",
            "Dispersed Flow": "#BA55D3",
            "Erratic/Noisy Flow": "#FF6347",
        }
        scatter_vol = px.scatter(
            vol_plot_df, x="Vol_Shannon", y="Vol_SampEn",
            color="VolRegimeName",
            color_discrete_map=color_map_vol,
            hover_data=["Close", "Volume"],
            labels={"Vol_Shannon": "Shannon Entropy (Concentration)", "Vol_SampEn": "Sample Entropy (Impulse Regularity)"},
        )
        scatter_vol.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="Volume Regime",
            margin=dict(l=20, r=20, b=20, t=20), height=450,
        )
        st.plotly_chart(scatter_vol, use_container_width=True)
    else:
        st.info(T("Volume Entropy data requires minimum 60 trading days.", "Du lieu Volume Entropy can toi thieu 60 ngay giao dich."))

# ==============================================================================
# AGENT DIAGNOSTIC -- TRI-VECTOR COMPOSITE SYNTHESIS
# ==============================================================================
st.markdown("---")
st.subheader(T("3. TRI-VECTOR COMPOSITE RISK DIAGNOSTIC", "3. CHẨN ĐOÁN RỦI RO HỢP THÀNH TRI-VECTOR"))
st.markdown(T(
    "Integrated composite diagnostic: 0.4*V1(Price) + 0.4*V2(Volume) + 0.2*V3(VN30 Breadth) scaled via Z-score + Sigmoid.",
    "Chẩn đoán hợp thành: 0.4*V1(Giá) + 0.4*V2(Thanh khoản) + 0.2*V3(Độ rộng VN30) chuẩn hóa Z-score + Sigmoid."
))

# Extract values for display
current_cse_norm = current_cse / 100.0 if pd.notna(current_cse) else 0.5
agent_regime = str(latest.get("RegimeName", "Calculating...")).upper()
current_vol_regime_upper = str(latest.get("VolRegimeName", "Calculating...")).upper()

# Structural Breadth analysis
if current_cse > 60:
    vn30_analysis = "High VN30 Entropy: Blue chips exhibit extreme fragmentation with no market consensus. Capital dispersion across sectors signals systemic decorrelation."
elif current_cse < 40:
    vn30_analysis = "Low VN30 Entropy: Blue chips maintain deterministic consensus. Centralized capital flow indicates institutional structural integrity."
else:
    vn30_analysis = "Neutral VN30 Entropy: Moderate internal rotation among capital pillars. Sector rebalancing underway without systemic stress."

vn30_breadth_status = "Unified" if current_cse < 40 else ("Divergent" if current_cse > 70 else "Fragmenting")

# Formatted code values
wpe_formatted = f"<code>{current_wpe:.4f}</code>"
flux_formatted = f"<code>{current_flux:+.3f}%</code>" if pd.notna(current_flux) else "<code>N/A</code>"
gz_formatted = f"<code>{current_vol_global_z:+.2f} Z</code>" if pd.notna(current_vol_global_z) else "<code>N/A</code>"
sh_formatted = f"<code>{vol_sh_kpi}</code>"
se_formatted = f"<code>{vol_se_kpi}</code>"
cse_formatted = f"<code>{current_cse_norm:.2f}</code>"
mfi_formatted = f"<code>{current_mfi:.4f}</code>"
risk_formatted = f"<code>{risk_score:.1f}</code>"

status_strong = f"<strong>{synthesis_label}</strong>"
breadth_strong = f"<strong>{vn30_breadth_status}</strong>"
regime_strong = f"<strong>{current_regime}</strong>"
vol_regime_strong = f"<strong>{current_vol_regime_upper}</strong>"
dominant_strong = f"<strong>{dominant_vector.replace('V1_Price','V1: PRICE').replace('V2_Volume','V2: VOLUME').replace('V3_Breadth','V3: BREADTH')}</strong>"

# Critical alert block
critical_block = f"""
<div class="critical-alert" style="border: 2px solid #FF3131; background: rgba(255, 49, 49, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
    <strong>[CRITICAL ALERT]</strong><br>
    Composite Risk Score ({risk_score:.1f}) exceeds P90 dynamic threshold ({critical_bound:.1f}). Phase transition imminent. 
    Dominant risk vector: {dominant_strong}. Immediate risk hedging recommended.
</div>
""" if "CRITICAL" in synthesis_label else ""

# Protocol logs
protocol_logs = f"""
> INITIATING TRI-VECTOR COMPOSITE ENGINE...<br>
> FITTING STANDARDIZED SHOCK SPACE (PowerTransform + Tied GMM)...<br>
> EXTRACTING VECTORS [V1: PRICE, V2: VOLUME, V3: BREADTH]...<br>
> NORMALIZING VIA PowerTransformer(yeo-johnson) + MinMaxScaler PIPELINE...<br>
> DYNAMIC RISK THRESHOLDS: ELEVATED = P75({elevated_bound:.1f}), CRITICAL = P90({critical_bound:.1f})<br>
> WEIGHTED SYNTHESIS COMPLETE (40/40/20 MODEL).<br>
"""

# Flux analysis
flux_val = current_flux if pd.notna(current_flux) else 0.0
if abs(flux_val) > 2.0:
    flux_analysis = "Entropy momentum is in a high-energy state, indicating rapid structural change."
elif abs(flux_val) > 0.5:
    flux_analysis = "Moderate entropy flux detected. Structural transition is underway but controlled."
else:
    flux_analysis = "Entropy momentum is near equilibrium. Structural stability maintained."

# Liquidity synthesis
gz_val_safe = current_vol_global_z if pd.notna(current_vol_global_z) else 0.0
vol_is_consensus = "CONSENSUS" in current_vol_regime_upper
if (gz_val_safe > 0 and latest['Close'] > df['SMA20'].iloc[-1]) or (gz_val_safe < 0 and latest['Close'] < df['SMA20'].iloc[-1]):
    liquidity_direction = "supporting"
else:
    liquidity_direction = "diverging from"

agent_log = f"""
<div class="agent-log">
{protocol_logs}
<br>
<h3>[ {T("TRI-VECTOR COMPOSITE RISK DIAGNOSTIC", "CHẨN ĐOÁN RỦI RO HỢP THÀNH TRI-VECTOR")} ]</h3>

| Vector Module | Key Metrics | Scaled Value | Weight |
| :--- | :--- | :--- | :--- |
| **V1: Price Entropy** | WPE: {wpe_formatted} -- Flux: {flux_formatted} | <code>{contributions.get('V1_Price',0):.4f}</code> | **40%** |
| **V2: Volume Entropy** | SampEn: {se_formatted} -- Macro Z: {gz_formatted} -- Shannon: {sh_formatted} | <code>{contributions.get('V2_Volume',0):.4f}</code> | **40%** |
| **V3: VN30 Breadth** | Corr Entropy: {cse_formatted} -- MFI: {mfi_formatted} | <code>{contributions.get('V3_Breadth',0):.4f}</code> | **20%** |
| **Composite Result** | Score: {risk_formatted}/100 -- Dominant: {dominant_strong} | -- | {status_strong} |

{critical_block}

<h3>[ {T("ANALYSIS", "PHAN TICH CHUYEN SAU")} ]</h3>
**1. Price Entropy Dynamics.** WPE at {wpe_formatted} places the market in the {regime_strong} regime (Tied GMM Topological Slicing). Momentum Entropy Flux at {flux_formatted} indicates {'entropy heating (destabilizing)' if flux_val > 0 else 'entropy cooling (stabilizing)'}. {flux_analysis}

**2. Volume-Price Fusion.** Current Liquidity Depth (Macro Z: {gz_formatted}) is {liquidity_direction} the current Price Regime ({regime_strong}). Volume structure is classified as {vol_regime_strong}. {'Liquidity is structurally aligned with price dynamics.' if vol_is_consensus else 'Structural fragility detected: volume behavior diverges from price entropy.'}

**3. Structural Breadth.** The Correlation Entropy (EVD) at {cse_formatted} identifies a {breadth_strong} internal structure. {vn30_analysis}

<h3>[ {T("CONCLUSION", "KẾT LUẬN")} ]</h3>
The Tri-Vector Composite engine reports a Risk Score of {risk_formatted}/100 ({status_strong}). The dominant risk contributor is {dominant_strong}. Market structural coherence is currently {'intact' if risk_score < 40 else 'under stress' if risk_score < 75 else 'critically degraded'}. The 40/40/20 weighted model confirms that {'systemic stability holds' if risk_score < 50 else 'structural divergence is active'}.

</div>
"""
st.markdown(agent_log, unsafe_allow_html=True)


# ==============================================================================
# DATA EXPORT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('2. DATA EXPORT', '2. XUẤT DỮ LIỆU')}**")
csv_data = df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label=T("Export Current Analysis (CSV)", "Xuất Dữ Liệu Hiện Tại (CSV)"),
    data=csv_data,
    file_name="financial_entropy_agent_export.csv",
    mime="text/csv",
    use_container_width=True
)
