import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_uae_pipeline_telemetry(steps=8760, seed=42):
    np.random.seed(seed)
    
    # 1. Generate Hourly Timestamps for a Full Calendar Year
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(steps)]
    
    t = np.arange(steps)
    
    # 2. Temperature: Diurnal (24h) + Seasonal (365 days) for Abu Dhabi Climate
    seasonal_temp = 32.0 + 12.0 * np.sin(2 * np.pi * (t - 1200) / 8760)  # Peak in Jul/Aug (~44°C)
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (t - 6) / 24)
    temperature_C = np.clip(seasonal_temp + diurnal_temp + np.random.normal(0, 1.2, steps), 15.0, 52.0)
    
    # 3. Relative Humidity: Coastal Swings + Critical Micro-Environment Trapping (11%-23%)
    seasonal_rh = 60.0 - 15.0 * np.sin(2 * np.pi * (t - 1200) / 8760)
    diurnal_rh = -20.0 * np.sin(2 * np.pi * (t - 6) / 24)
    humidity_pct = np.clip(seasonal_rh + diurnal_rh + np.random.normal(0, 3.0, steps), 12.0, 98.0)
    
    # 4. Salinity (Arabian Gulf Baseline: 42-45 g/L)
    salinity_gL = np.clip(43.5 + 1.5 * np.sin(2 * np.pi * t / 4380) + np.random.normal(0, 0.4, steps), 38.0, 48.0)
    
    # 5. Sulfate-Reducing Bacteria (SRB) Multiplier (1.0x to 4.0x)
    srb_base = 1.0 + 0.3 * np.sin(2 * np.pi * t / 720)
    srb_spikes = np.where(np.random.rand(steps) > 0.96, np.random.uniform(1.8, 3.5, steps), 1.0)
    srb_multiplier = np.clip(srb_base * srb_spikes, 1.0, 4.0)
    
    # 6. Physical Corrosion Rate Calculation (Arrhenius Kinetics + Regional Baseline 0.056 mm/yr)
    Ea = 38000.0  # J/mol
    R = 8.314     # J/(mol*K)
    T_kelvin = temperature_C + 273.15
    arrhenius_factor = np.exp(-Ea / (R * T_kelvin)) / np.exp(-Ea / (R * 298.15))
    
    # Critical RH salt-crust trapping factor
    rh_factor = np.where(humidity_pct > 11.0, 1.0 + (humidity_pct / 100.0) * 0.25, 0.85)
    
    # Baseline coastal steel corrosion: 0.056 mm/year
    base_rate_mm_yr = 0.056
    corrosion_rate = base_rate_mm_yr * arrhenius_factor * rh_factor * (salinity_gL / 43.5) * srb_multiplier
    
    # Cumulative Corrosion Depth (mm)
    corrosion_depth_mm = np.cumsum(corrosion_rate / 8760.0)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'month': [ts.strftime('%B') for ts in timestamps],
        'month_num': [ts.month for ts in timestamps],
        'temperature_C': temperature_C,
        'humidity_pct': humidity_pct,
        'salinity_gL': salinity_gL,
        'srb_multiplier': srb_multiplier,
        'corrosion_rate': corrosion_rate,
        'corrosion_depth_mm': corrosion_depth_mm
    })
    
    df.to_csv("data/simulated_corrosion.csv", index=False)
    print(f"✅ Generated {steps} rows of UAE-calibrated telemetry in data/simulated_corrosion.csv")
    return df

if __name__ == "__main__":
    generate_uae_pipeline_telemetry()