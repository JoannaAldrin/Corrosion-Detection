"""
Gulf-calibrated synthetic corrosion simulator.
Two physically distinct mechanisms:
  - EXTERNAL corrosion: ambient temperature, soil/seawater salinity, and pH
    acting on the outer pipe wall (Arrhenius-based) [1][2][5]
  - INTERNAL corrosion: sulfate-reducing bacteria (MIC) acting on the inner
    wall, tied to the fluid flowing through the pipe -- water cut, flow
    velocity, and internal fluid temperature [6]
Also tags every row with its calendar month and Gulf season (hot/mild) so
the dataset can be segregated for seasonal analysis.
"""
import numpy as np
import pandas as pd
import datetime
import os

A = 3.0e5
EA = 45_000.0
R = 8.314

def arrhenius_rate(temperature_c):
    temperature_k = temperature_c + 273.15
    return A * np.exp(-EA / (R * temperature_k))

def salinity_factor(salinity_pct):
    return 1.0 + 0.15 * salinity_pct

def ph_factor(ph):
    return 1.0 + 0.08 * np.abs(ph - 7.0)

def external_corrosion_rate(temperature_c, salinity_pct, ph, rng):
    """Outer-wall corrosion -- ambient/environmental drivers."""
    rate = arrhenius_rate(temperature_c) * salinity_factor(salinity_pct) * ph_factor(ph)
    return np.clip(rate + rng.normal(0, rate.std() * 0.1, rate.shape), 0, None)

def internal_corrosion_rate(internal_temp_c, water_cut_pct, flow_velocity_m_s, rng):
    """
    Inner-wall corrosion -- MIC/SRB driven. SRB activity is favored by
    moderate temperature (20-40 C), low flow velocity (stagnant sections
    let biofilms settle), and higher water cut [6].
    """
    srb_favorable = (
        (internal_temp_c > 20) & (internal_temp_c < 40)
        & (flow_velocity_m_s < 0.5) & (water_cut_pct > 20)
    )
    base_rate = 0.01
    boost = rng.uniform(2.0, 6.0, size=internal_temp_c.shape)
    rate = base_rate * np.where(srb_favorable, boost, 1.0) * (1 + water_cut_pct / 100)
    return np.clip(rate + rng.normal(0, rate.std() * 0.1, rate.shape), 0, None)

def month_and_season(day_of_sim, start_date=datetime.date(2024, 1, 1)):
    """Tags each simulated day with its calendar month and Gulf season.
    UAE/Gulf climate: hot season May-Sep, mild season Oct-Apr."""
    dates = [start_date + datetime.timedelta(days=int(d)) for d in day_of_sim]
    months = np.array([d.month for d in dates])
    seasons = np.where((months >= 5) & (months <= 9), "hot", "mild")
    return months, seasons

def simulate_pipe(pipe_id, n_days, rng, climate):
    day = np.arange(n_days)
    month, season = month_and_season(day)

    # --- External/environmental conditions (outer wall) ---
    if climate == "desert":
        temperature_c = 32 + 12 * np.sin(2*np.pi*day/365) + rng.normal(0, 2.5, n_days)
        humidity_pct = np.clip(35 + 15*np.sin(2*np.pi*day/365+1) + rng.normal(0,5,n_days), 5, 100)
        salinity_pct = np.clip(rng.normal(0.02, 0.005, n_days), 0.005, 0.05)
        ph = np.clip(rng.normal(7.4, 0.3, n_days), 5.5, 9.0)
    else:  # subsea, calibrated to Gulf seawater [5]
        temperature_c = 27 + 4 * np.sin(2*np.pi*day/365) + rng.normal(0, 1.0, n_days)
        humidity_pct = np.full(n_days, 100.0)
        salinity_pct = np.clip(rng.normal(0.040, 0.003, n_days), 0.03, 0.05)
        ph = np.clip(rng.normal(8.1, 0.15, n_days), 7.5, 8.5)

    # --- Internal/fluid conditions (inner wall, independent of climate) ---
    water_cut_pct = np.clip(30 + 20*np.sin(2*np.pi*day/180) + rng.normal(0, 8, n_days), 0, 95)
    flow_velocity_m_s = np.clip(rng.normal(1.5, 0.8, n_days), 0.05, 4.0)
    internal_temp_c = np.clip(45 + rng.normal(0, 4, n_days), 15, 75)

    ext_daily = external_corrosion_rate(temperature_c, salinity_pct, ph, rng)
    int_daily = internal_corrosion_rate(internal_temp_c, water_cut_pct, flow_velocity_m_s, rng)

    return pd.DataFrame({
        "pipe_id": pipe_id, "climate": climate, "day": day,
        "month": month, "season": season,
        "temperature_C": temperature_c, "humidity_pct": humidity_pct,
        "salinity_pct": salinity_pct, "ph": ph,
        "water_cut_pct": water_cut_pct, "flow_velocity_m_s": flow_velocity_m_s,
        "internal_temp_C": internal_temp_c,
        "external_daily_rate": ext_daily, "internal_daily_rate": int_daily,
        "external_corrosion_depth_mm": np.cumsum(ext_daily),
        "internal_corrosion_depth_mm": np.cumsum(int_daily),
    })

def main(n_pipes=40, n_days=730, seed=42, out_path="data/simulated_corrosion.csv"):
    rng = np.random.default_rng(seed)
    frames = [simulate_pipe(i, n_days, rng, "desert" if i % 2 == 0 else "subsea")
              for i in range(n_pipes)]
    df = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {n_pipes} pipes to {out_path}")

    monthly_summary = (
        df.groupby(["climate", "month", "season"])[
            ["temperature_C", "external_daily_rate", "internal_daily_rate"]
        ].mean().reset_index()
    )
    monthly_summary.to_csv("data/monthly_summary.csv", index=False)
    print("Wrote data/monthly_summary.csv -- mean conditions and corrosion rate per month")

if __name__ == "__main__":
    main()