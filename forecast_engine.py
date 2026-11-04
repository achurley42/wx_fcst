# forecast_engine.py
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import shift
from datetime import datetime, timedelta

SURFACE_KEYS = ["t2m", "d2m", "u10", "v10", "msl", "tp"]


def run_72hr_forecast(grid_data, lats, lons, model, compute_mgr, start_time_utc, weights_path=None):
    """
    Executes a 12-step (6-hourly) autoregressive rollout to T+72 hours.
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[MODEL] Loading fine-tuned weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("[MODEL] No custom weights specified. Running Physics-Guided Neural Hybrid baseline.")
    model.eval()

    nlat, nlon = len(lats), len(lons)
    orog = grid_data["orog"]
    lsm = grid_data["lsm"]

    # Normalize static features
    orog_norm = (orog - np.mean(orog)) / (np.std(orog) + 1e-5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Storage for 13 time steps (T+00h to T+72h at 6-hr intervals)
    history = {k: np.zeros((13, nlat, nlon), dtype=np.float32) for k in SURFACE_KEYS}
    for k in SURFACE_KEYS:
        history[k][0] = grid_data[k]

    valid_times = [start_time_utc + timedelta(hours=6 * step) for step in range(13)]

    # Extract 850 hPa steering winds for synoptic advection coupling
    u_steer = grid_data.get("u_850", grid_data["u10"] * 1.5)
    v_steer = grid_data.get("v_850", grid_data["v10"] * 1.5)
    res_deg = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25
    dx_meters = res_deg * 111000.0

    for step in range(1, 13):
        fhr = step * 6
        vtime = valid_times[step]

        # 1. Diurnal Solar Forcing Angle (peaks at 14:00 Local Solar Time)
        utc_hr = vtime.hour
        local_solar_hr = (utc_hr + lon_grid / 15.0) % 24.0
        diurnal_wave = np.cos((local_solar_hr - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        prev_diurnal = np.cos(((utc_hr - 6.0 + lon_grid / 15.0) % 24.0 - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        diurnal_delta = diurnal_wave - prev_diurnal

        # 2. Assemble 30-Channel Input Tensor for U-Net
        channels = []
        for k in SURFACE_KEYS:
            arr = history[k][step - 1]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        for k in [key for key in grid_data.keys() if "_" in key]:
            arr = grid_data[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        channels.extend([orog_norm, lsm, diurnal_wave, np.sin( np.deg2rad(lat_grid) ).astype(np.float32)])

        # Pad or trim to 30 channels
        while len(channels) < 30:
            channels.append(np.zeros_like(orog_norm))
        x_tensor = torch.tensor(np.stack(channels[:30], axis=0), dtype=torch.float32).unsqueeze(0)

        # 3. Hybrid GPU/CPU Neural Residual Inference
        nn_delta = compute_mgr.safe_forward(model, x_tensor).squeeze(0).numpy()

        # 4. Combine Neural Residual + Synoptic Advection + Topographic/Diurnal Physics
        shift_x = float(np.mean(u_steer) * 21600.0 / dx_meters) * 0.35
        shift_y = float(np.mean(v_steer) * 21600.0 / dx_meters) * 0.35

        prev_t2m = shift(history["t2m"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_d2m = shift(history["d2m"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_msl = shift(history["msl"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_u10 = shift(history["u10"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_v10 = shift(history["v10"][step - 1], shift=(shift_y, shift_x), mode="nearest")

        # Land heats/cools 3.5x stronger than ocean/buoy points (controlled by LSM)
        diurnal_amp = 0.8 + 3.2 * lsm
        history["t2m"][step] = prev_t2m + diurnal_amp * diurnal_delta + nn_delta[0] * 0.5
        # Enforce physical rule: Dewpoint cannot exceed Temperature (RH <= 100%)
        history["d2m"][step] = np.minimum(prev_d2m + 0.25 * diurnal_amp * diurnal_delta + nn_delta[1] * 0.3, history["t2m"][step])
        history["u10"][step] = prev_u10 + nn_delta[2] * 0.4 - (0.05 * lsm * prev_u10)
        history["v10"][step] = prev_v10 + nn_delta[3] * 0.4 - (0.05 * lsm * prev_v10)
        history["msl"][step] = prev_msl + nn_delta[4] * 50.0

        # Diagnostic 6-hr precipitation from low-level moisture convergence & relative humidity
        rh_approx = np.clip(100.0 - 5.0 * (history["t2m"][step] - history["d2m"][step]), 5.0, 100.0)
        orog_lift = np.clip(orog / 1000.0, 0.0, 2.5)
        precip_6h = np.where(rh_approx > 80.0, ((rh_approx - 80.0) / 20.0)**2 * (3.5 + orog_lift) + np.abs(nn_delta[5]), 0.0)
        history["tp"][step] = np.clip(precip_6h, 0.0, 75.0)

        print(f"  -> Completed Forecast Step T+{fhr:02d}h ({vtime.strftime('%Y-%m-%d %H:%MZ')})")

    return history, valid_times


def export_deliverables(history, lats, lons, valid_times, orog, lsm, output_dir):
    """Exports 72-hr NetCDF4 cube, Center-Point CSV Meteogram, and 4-Panel PNG Surface Charts."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export Gridded NetCDF4
    nc_path = os.path.join(output_dir, "forecast_72hr_surface.nc")
    ds_out = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], history["t2m"] - 273.15, {"units": "degC", "long_name": "2m Temperature"}),
            "d2m": (["time", "lat", "lon"], history["d2m"] - 273.15, {"units": "degC", "long_name": "2m Dewpoint"}),
            "u10": (["time", "lat", "lon"], history["u10"], {"units": "m/s", "long_name": "10m U-Wind"}),
            "v10": (["time", "lat", "lon"], history["v10"], {"units": "m/s", "long_name": "10m V-Wind"}),
            "msl": (["time", "lat", "lon"], history["msl"] / 100.0, {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
            "tp":  (["time", "lat", "lon"], history["tp"], {"units": "mm", "long_name": "6-Hr Accumulated Precipitation"}),
            "orog": (["lat", "lon"], orog, {"units": "m", "long_name": "Terrain Elevation"}),
            "lsm":  (["lat", "lon"], lsm, {"units": "fraction", "long_name": "Land-Sea Mask"}),
        },
        coords={"time": valid_times, "lat": lats, "lon": lons}
    )
    ds_out.to_netcdf(nc_path)
    print(f"[EXPORT] Saved 72-Hour Gridded NetCDF4: {nc_path}")

    # 2. Export Station / Grid-Center Summary CSV
    mid_lat_idx, mid_lon_idx = len(lats) // 2, len(lons) // 2
    df_rows = []
    for t_idx, vtime in enumerate(valid_times):
        u = history["u10"][t_idx, mid_lat_idx, mid_lon_idx]
        v = history["v10"][t_idx, mid_lat_idx, mid_lon_idx]
        wspd_kt = np.hypot(u, v) * 1.94384
        wdir = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0
        df_rows.append({
            "Forecast_Hour": f"T+{t_idx*6:02d}h",
            "Valid_Time_UTC": vtime.strftime("%Y-%m-%d %H:%MZ"),
            "Center_Lat": round(float(lats[mid_lat_idx]), 3),
            "Center_Lon": round(float(lons[mid_lon_idx]), 3),
            "Temp_2m_C": round(float(history["t2m"][t_idx, mid_lat_idx, mid_lon_idx] - 273.15), 1),
            "Dewpoint_2m_C": round(float(history["d2m"][t_idx, mid_lat_idx, mid_lon_idx] - 273.15), 1),
            "Wind_Dir_Deg": int(round(wdir)),
            "Wind_Spd_Kts": round(float(wspd_kt), 1),
            "MSLP_hPa": round(float(history["msl"][t_idx, mid_lat_idx, mid_lon_idx] / 100.0), 1),
            "Precip_6hr_mm": round(float(history["tp"][t_idx, mid_lat_idx, mid_lon_idx]), 2)
        })
    csv_path = os.path.join(output_dir, "aor_center_point_forecast.csv")
    pd.DataFrame(df_rows).to_csv(csv_path, index=False)
    print(f"[EXPORT] Saved Point Summary CSV: {csv_path}")

    # 3. Generate Automated 4-Panel Surface Prognostic Charts (T+24h, T+48h, T+72h)
    import matplotlib.pyplot as plt
    lon_g, lat_g = np.meshgrid(lons, lats)
    for step_idx in [4, 8, 12]:  # 24h, 48h, 72h
        fhr = step_idx * 6
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"AI Regional Surface Forecast | Valid: {valid_times[step_idx].strftime('%Y-%m-%d %H:%MZ')} (T+{fhr:02d}h)", fontsize=14, fontweight="bold")

        # Panel 1: 2m Temperature (C) & Terrain Contours
        c1 = axes[0, 0].contourf(lon_g, lat_g, history["t2m"][step_idx] - 273.15, levels=15, cmap="RdYlBu_r")
        axes[0, 0].contour(lon_g, lat_g, lsm, levels=[0.5], colors="black", linewidths=1.2)
        axes[0, 0].set_title("2m Temperature (°C) & Coastline/LSM")
        fig.colorbar(c1, ax=axes[0, 0])

        # Panel 2: MSLP (hPa) & 10m Wind Vectors
        msl_hpa = history["msl"][step_idx] / 100.0
        wspd_kt = np.hypot(history["u10"][step_idx], history["v10"][step_idx]) * 1.94384
        c2 = axes[0, 1].contourf(lon_g, lat_g, wspd_kt, levels=12, cmap="YlOrRd")
        cs = axes[0, 1].contour(lon_g, lat_g, msl_hpa, levels=10, colors="black", linewidths=0.8)
        axes[0, 1].clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
        skip = max(1, len(lats) // 12)
        axes[0, 1].barbs(lon_g[::skip, ::skip], lat_g[::skip, ::skip],
                         history["u10"][step_idx, ::skip, ::skip] * 1.94384,
                         history["v10"][step_idx, ::skip, ::skip] * 1.94384, length=5)
        axes[0, 1].set_title("MSLP (hPa, Contours) & 10m Wind Speed (kts)")
        fig.colorbar(c2, ax=axes[0, 1])

        # Panel 3: 2m Dewpoint (C)
        c3 = axes[1, 0].contourf(lon_g, lat_g, history["d2m"][step_idx] - 273.15, levels=15, cmap="BrBG")
        axes[1, 0].contour(lon_g, lat_g, lsm, levels=[0.5], colors="black", linewidths=1.2)
        axes[1, 0].set_title("2m Dewpoint Temperature (°C)")
        fig.colorbar(c3, ax=axes[1, 0])

        # Panel 4: 6-Hr Precipitation (mm) & Terrain Orography
        c4 = axes[1, 1].contourf(lon_g, lat_g, history["tp"][step_idx], levels=12, cmap="Blues")
        axes[1, 1].contour(lon_g, lat_g, orog, levels=6, colors="saddlebrown", linewidths=0.7, alpha=0.6)
        axes[1, 1].set_title("6-Hr Precip (mm) & Terrain Elevation (Brown Contours)")
        fig.colorbar(c4, ax=axes[1, 1])

        png_path = os.path.join(output_dir, f"surface_prog_T+{fhr:02d}h.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[EXPORT] Saved 4-Panel Surface Chart: {png_path}")
