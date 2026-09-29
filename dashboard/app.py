import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Abu Dhabi Pipeline Corrosion Digital Twin", layout="wide")

# Function to fetch live weather from Open-Meteo API (Abu Dhabi Coordinates)
@st.cache_data(ttl=600)
def get_live_abudhabi_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=24.4539&longitude=54.3773&current=temperature_2m,relative_humidity_2m"
        res = requests.get(url, timeout=5).json()
        temp = res['current']['temperature_2m']
        rh = res['current']['relative_humidity_2m']
        return temp, rh, True
    except Exception:
        return 38.5, 62.0, False # Fallback values

# Load Telemetry Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/simulated_corrosion.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

st.title("🛡️ Abu Dhabi Pipeline Corrosion Digital Twin")
st.markdown("Physics-Informed Monitoring & Risk Assessment Engine Calibrated to Arabian Gulf Field Studies.")

# Sidebar Live Weather Integration
live_temp, live_rh, weather_success = get_live_abudhabi_weather()
st.sidebar.header("🌐 Live Abu Dhabi Telemetry")
if weather_success:
    st.sidebar.success("Connected to Live Open-Meteo API")
else:
    st.sidebar.warning("Using Offline Calibration Metrics")

st.sidebar.metric("Live Ambient Temp (°C)", f"{live_temp:.1f} °C")
st.sidebar.metric("Live Relative Humidity (%)", f"{live_rh:.1f} %")

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📡 Automated Live Telemetry", 
    "📅 Monthly Risk Breakdown", 
    "🧪 What-If Scenario Planner"
])

# ==========================================
# TAB 1: AUTOMATED LIVE MONITORING
# ==========================================
with tab1:
    st.subheader("Automated Operational Early Warning Engine")
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Temp (°C)", f"{latest['temperature_C']:.1f}")
    col2.metric("Relative Humidity (%)", f"{latest['humidity_pct']:.1f}")
    col3.metric("Salinity (g/L)", f"{latest['salinity_gL']:.1f}")
    col4.metric("SRB Factor", f"{latest['srb_multiplier']:.2f}x")
    
    st.write("---")
    
    # Automated Risk Decision Logic
    rate = latest['corrosion_rate']
    srb = latest['srb_multiplier']
    
    if rate > 0.06 or srb > 2.5:
        st.error("🚨 **CRITICAL ALERT: Microbially Influenced Corrosion (MIC) Threat**")
        st.caption("High Sulfate-Reducing Bacteria (SRB) activity indicates severe localized pitting risks.")
    elif rate > 0.035:
        st.warning("⚠️ **WARNING: Elevated Thermal Oxidation Risk**")
        st.caption("Peak desert temperature and high salinity driving accelerated wall thinning.")
    else:
        st.success("✅ **SYSTEM HEALTHY: Operating Within Safe Physics Bounds**")
        st.caption("Degradation rates remain within baseline thresholds (<0.035 mm/year).")
        
    st.write("### 24-Hour Rolling Forecast")
    st.line_chart(df.set_index('timestamp')[['corrosion_rate', 'corrosion_depth_mm']].tail(24))

# ==========================================
# TAB 2: MONTHLY CORROSION RISK BREAKDOWN
# ==========================================
with tab2:
    st.subheader("📅 Calendar Month Risk Segregation (Professor Feedback Implementation)")
    st.markdown("Aggregation of 8,760 hourly timesteps into calendar months to identify seasonal risk windows in the UAE.")
    
    # Group by month maintaining chronological order
    monthly_agg = df.groupby(['month_num', 'month'], as_index=False).agg({
        'corrosion_rate': 'mean',
        'temperature_C': 'mean',
        'humidity_pct': 'mean',
        'srb_multiplier': 'mean'
    }).sort_values('month_num')
    
    # Highlight Peak Month
    peak_row = monthly_agg.loc[monthly_agg['corrosion_rate'].idxmax()]
    
    st.info(f"🔥 **Peak Risk Period:** **{peak_row['month']}** exhibits the highest mean corrosion rate (**{peak_row['corrosion_rate']:.4f} mm/yr**) driven by mean temperatures of **{peak_row['temperature_C']:.1f}°C**.")
    
    # Monthly Bar Chart
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    colors = ['#ff4b4b' if m == peak_row['month'] else '#1f77b4' for m in monthly_agg['month']]
    ax1.bar(monthly_agg['month'], monthly_agg['corrosion_rate'], color=colors, alpha=0.8)
    ax1.set_ylabel("Mean Corrosion Rate (mm/yr)", color="#1f77b4")
    plt.xticks(rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(monthly_agg['month'], monthly_agg['temperature_C'], color='orange', marker='o', linewidth=2, label="Temp (°C)")
    ax2.set_ylabel("Mean Temp (°C)", color="orange")
    
    st.pyplot(fig)
    
    # Data Table
    st.write("### Monthly Environmental Summary Table")
    st.dataframe(monthly_agg[['month', 'corrosion_rate', 'temperature_C', 'humidity_pct', 'srb_multiplier']].style.highlight_max(subset=['corrosion_rate'], color='#ffcccc'))

# ==========================================
# TAB 3: WHAT-IF SCENARIO PLANNER
# ==========================================
with tab3:
    st.subheader("🧪 Stress Testing & Scenario Planner")
    st.caption("Manually adjust environmental inputs to simulate extreme out-of-bounds desert heat or microbial blooms.")
    
    cA, cB = st.columns(2)
    with cA:
        s_temp = st.slider("Simulated Temp (°C)", 15.0, 55.0, 45.0)
        s_rh = st.slider("Simulated Humidity (%)", 10.0, 100.0, 75.0)
    with cB:
        s_sal = st.slider("Simulated Salinity (g/L)", 30.0, 50.0, 44.0)
        s_srb = st.slider("Simulated SRB Multiplier", 1.0, 4.0, 2.8)
        
    # Arrhenius Calculation
    Ea, R = 38000.0, 8.314
    arr = np.exp(-Ea / (R * (s_temp + 273.15))) / np.exp(-Ea / (R * 298.15))
    sim_rate = 0.056 * arr * (1.0 + (s_rh / 100.0) * 0.25) * (s_sal / 43.5) * s_srb
    
    st.metric("Simulated Corrosion Rate", f"{sim_rate:.4f} mm/year")