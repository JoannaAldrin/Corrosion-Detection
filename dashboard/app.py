import streamlit as st
import torch, numpy as np, pandas as pd, requests, sys, os
import plotly.graph_objects as go
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from common import CorrosionLSTM, ALL_FEATURES

st.set_page_config(page_title="Gulf Corrosion Twin", page_icon="⚠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #0f2b36 0%, #081418 55%, #060c0e 100%); color: #E8F0F2; }
section[data-testid="stSidebar"] { background: #0A1C22; border-right: 1px solid #16343D; }
.kpi-card { background: linear-gradient(145deg, #0E2A33, #0A1E24); border: 1px solid #1C4450;
  border-radius: 14px; padding: 20px 22px; box-shadow: 0 4px 18px rgba(0,0,0,0.35); }
.kpi-label { font-size: 11px; letter-spacing: 2px; color: #6FA8B5; text-transform: uppercase; font-weight: 600; }
.kpi-value { font-size: 30px; font-weight: 800; color: #F4F7F8; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
.section-tag { font-size: 12px; letter-spacing: 2px; color: #E4572E; font-weight: 700; text-transform: uppercase; margin: 18px 0 6px 0; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge-low { background: rgba(39,194,139,0.15); color: #27C28B; border: 1px solid #27C28B; }
.badge-moderate { background: rgba(230,183,60,0.15); color: #E6B73C; border: 1px solid #E6B73C; }
.badge-elevated { background: rgba(228,87,46,0.15); color: #E4572E; border: 1px solid #E4572E; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = CorrosionLSTM()
    model.load_state_dict(torch.load("model/pinn_lstm_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data(ttl=300)  # 5 min cache -- short enough to stay "live"
def fetch_live_weather(lat=25.2048, lon=55.2708):  # Dubai coordinates
    """
    Uses Open-Meteo's `current=` parameter, which returns the actual
    current reading directly -- NOT the hourly forecast array. Indexing
    the hourly array (e.g. temperature_2m[-1]) returns a forecasted value
    up to 48 hours out, which is the bug that caused earlier mismatches.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m&timezone=auto"
    )
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    current = r.json()["current"]
    return current["temperature_2m"], current["relative_humidity_2m"]

def risk_badge(rate):
    if rate < 0.15: return "LOW", "low"
    if rate < 0.35: return "MODERATE", "moderate"
    return "ELEVATED", "elevated"

st.title("Gulf Corrosion Digital Twin")
st.caption("Separate early-warning tracks for external (environmental) and internal (microbial) corrosion")

with st.sidebar:
    st.markdown('<div class="section-tag">External / Surface Conditions</div>', unsafe_allow_html=True)
    live = st.toggle("Enable Live Telemetry Feed", value=True)
    if live:
        try:
            temp, hum = fetch_live_weather()
            st.success(f"Live: {temp:.1f}°C / {hum:.0f}% RH")
        except Exception as e:
            st.warning(f"Live feed unavailable ({e}) — using manual input.")
            live = False
    if not live:
        temp = st.slider("Temperature (°C)", 10.0, 55.0, 34.0)
        hum = st.slider("Humidity (%)", 0.0, 100.0, 45.0)
    salinity = st.slider("Soil / Water Salinity (%)", 0.5, 5.0, 2.5) / 100
    ph = st.slider("pH", 5.5, 9.0, 7.4)

    st.markdown('<div class="section-tag">Internal / Fluid Conditions</div>', unsafe_allow_html=True)
    internal_temp = st.slider("Internal Fluid Temp (°C)", 15.0, 75.0, 45.0)
    water_cut = st.slider("Water Cut (%)", 0.0, 95.0, 30.0)
    flow_velocity = st.slider("Flow Velocity (m/s)", 0.05, 4.0, 1.5)

window = np.tile(
    [temp, hum, salinity, ph, internal_temp, water_cut, flow_velocity], (30, 1)
).astype(np.float32)
x = torch.from_numpy(window).unsqueeze(0)

model = load_model()
with torch.no_grad():
    pred = model(x).squeeze(0).numpy()
ext_depth, int_depth = float(pred[0]), float(pred[1])
ext_rate, int_rate = ext_depth / 2.0, int_depth / 2.0

st.markdown('<div class="section-tag">External Corrosion (Surface)</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
level, cls = risk_badge(ext_rate)
c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Predicted Depth</div><div class="kpi-value">{ext_depth:.3f} mm</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Annual Rate</div><div class="kpi-value">{ext_rate:.3f} mm/yr</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Risk Rating</div><div class="badge badge-{cls}">{level}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-tag">Internal Corrosion (Microbial / MIC)</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
level2, cls2 = risk_badge(int_rate)
c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Predicted Depth</div><div class="kpi-value">{int_depth:.3f} mm</div></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="kpi-card"><div class="kpi-label">Annual Rate</div><div class="kpi-value">{int_rate:.3f} mm/yr</div></div>', unsafe_allow_html=True)
c6.markdown(f'<div class="kpi-card"><div class="kpi-label">Risk Rating</div><div class="badge badge-{cls2}">{level2}</div></div>', unsafe_allow_html=True)

st.markdown("---")
days = np.arange(0, 365)
fig = go.Figure()
fig.add_trace(go.Scatter(x=days, y=ext_depth*(days/30)**0.9, name="External", line=dict(color="#6FA8B5", width=3)))
fig.add_trace(go.Scatter(x=days, y=int_depth*(days/30)**0.9, name="Internal (MIC)", line=dict(color="#E4572E", width=3)))
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   height=380, xaxis_title="Days ahead", yaxis_title="Corrosion depth (mm)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)