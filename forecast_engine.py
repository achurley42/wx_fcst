# forecast_engine.py
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import shift
from datetime import datetime, timedelta

SURFACE_KEYS = ["t2m", "d2m", "u10", "v10", "msl", "tp"]


def run_hourly_72hr_forecast(grid_data, lats, lons, model, compute_mgr, start_time_utc, weights_path=None):
    """
    Executes a 72-hour autoregressive forecast at 1-hour intervals (73 time steps: T+00h to T+72h).
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[MODEL] Loading fine-tuned weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("[MODEL] No custom weights specified. Running Physics-Guided Neural Hybrid baseline (Hourly Mode).")
    model.eval()

    nlat, nlon = len(lats), len(lons)
    orog = grid_data["orog"]
    lsm = grid_data["lsm"]

    # Normalize static orography
    orog_norm = (orog - np.mean(orog)) / (np.std(orog) + 1e-5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Pre-allocate arrays for 73 hourly steps (T+00h to T+72h)
    num_steps = 73
    history = {k: np.zeros((num_steps, nlat, nlon), dtype=np.float32) for k in SURFACE_KEYS}
    for k in SURFACE_KEYS:
        history[k][0] = grid_data[k]

    valid_times = [start_time_utc + timedelta(hours=step) for step in range(num_steps)]

    # Extract 850 hPa steering winds
    u_steer = grid_data.get("u_850", grid_data["u10"] * 1.5)
    v_steer = grid_data.get("v_850", grid_data["v10"] * 1.5)

    res_deg = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
    dx_meters = res_deg * 111000.0

    # Hourly advection displacement (dt = 3600 seconds)
    shift_x = float(np.mean(u_steer) * 3600.0 / dx_meters) * 0.35
    shift_y = float(np.mean(v_steer) * 3600.0 / dx_meters) * 0.35

    print(f"[FORECAST] Initiating 72-hour rollout at 1-hour cadence ({num_steps} total states)...")

    for step in range(1, num_steps):
        vtime = valid_times[step]
        utc_hr = vtime.hour

        # 1. Hourly Astronomical Diurnal Cycle (Solar elevation relative to 14:00 local peak)
        local_solar_hr = (utc_hr + lon_grid / 15.0) % 24.0
        diurnal_wave = np.cos((local_solar_hr - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)

        prev_utc_hr = (utc_hr - 1.0) % 24.0
        prev_solar_hr = (prev_utc_hr + lon_grid / 15.0) % 24.0
        prev_diurnal = np.cos((prev_solar_hr - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        diurnal_delta = diurnal_wave - prev_diurnal

        # 2. Assemble 30-channel input tensor
        channels = []
        for k in SURFACE_KEYS:
            arr = history[k][step - 1]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        for k in [key for key in grid_data.keys() if "_" in key]:
            arr = grid_data[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        channels.extend([
            orog_norm,
            lsm,
            diurnal_wave,
            np.sin(np.deg2rad(lat_grid)).astype(np.float32)
        ])

        while len(channels) < 30:
            channels.append(np.zeros_like(orog_norm))

        x_tensor = torch.tensor(np.stack(channels[:30], axis=0), dtype=torch.float32).unsqueeze(0)

        # 3. Neural residual inference (scaled to 1-hour tendency)
        nn_delta = compute_mgr.safe_forward(model, x_tensor).squeeze(0).numpy() / 6.0

        # 4. Hourly physical advection shift
        prev_t2m = shift(history["t2m"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_d2m = shift(history["d2m"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_msl = shift(history["msl"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_u10 = shift(history["u10"][step - 1], shift=(shift_y, shift_x), mode="nearest")
        prev_v10 = shift(history["v10"][step - 1], shift=(shift_y, shift_x), mode="nearest")

        # 5. Land-Atmosphere coupling & diurnal heating rate
        diurnal_amp = 0.8 + 3.2 * lsm
        history["t2m"][step] = prev_t2m + diurnal_amp * diurnal_delta + nn_delta[0] * 0.5

        # Enforce Td <= T (RH <= 100%)
        history["d2m"][step] = np.minimum(
            prev_d2m + 0.25 * diurnal_amp * diurnal_delta + nn_delta[1] * 0.3,
            history["t2m"][step]
        )

        history["u10"][step] = prev_u10 + nn_delta[2] * 0.4 - (0.01 * lsm * prev_u10)
        history["v10"][step] = prev_v10 + nn_delta[3] * 0.4 - (0.01 * lsm * prev_v10)
        history["msl"][step] = prev_msl + nn_delta[4] * 10.0

        # 6. Diagnostic 1-hour accumulated precipitation
        rh_approx = np.clip(100.0 - 5.0 * (history["t2m"][step] - history["d2m"][step]), 5.0, 100.0)
        orog_lift = np.clip(orog / 1000.0, 0.0, 2.5)
        precip_1h = np.where(
            rh_approx > 85.0,
            ((rh_approx - 85.0) / 15.0)**2 * (0.8 + orog_lift * 0.3) + np.abs(nn_delta[5]),
            0.0
        )
        history["tp"][step] = np.clip(precip_1h, 0.0, 30.0)

        if step % 6 == 0 or step == num_steps - 1:
            print(f"  -> Generated Forecast Hour T+{step:02d}h ({vtime.strftime('%Y-%m-%d %H:%MZ')})")

    return history, valid_times


def export_hourly_deliverables(history, lats, lons, valid_times, orog, lsm, output_dir, plot_interval=6):
    """
    Exports individual hourly NetCDF4 files, master 73-step NetCDF cube,
    73-step meteogram CSV, and surface prognostic charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    hourly_nc_dir = os.path.join(output_dir, "hourly_nc")
    os.makedirs(hourly_nc_dir, exist_ok=True)

    num_steps = len(valid_times)

    # 1. Export Individual Hourly NetCDF4 Files
    print(f"[EXPORT] Writing {num_steps} individual hourly NetCDF4 files to '{hourly_nc_dir}'...")
    for step_idx, vtime in enumerate(valid_times):
        hourly_file_path = os.path.join(hourly_nc_dir, f"forecast_surface_T+{step_idx:02d}h.nc")
        ds_hourly = xr.Dataset(
            {
                "t2m": (["lat", "lon"], history["t2m"][step_idx] - 273.15, {"units": "degC", "long_name": "2m Temperature"}),
                "d2m": (["lat", "lon"], history["d2m"][step_idx] - 273.15, {"units": "degC", "long_name": "2m Dewpoint"}),
                "u10": (["lat", "lon"], history["u10"][step_idx], {"units": "m/s", "long_name": "10m U-Wind"}),
                "v10": (["lat", "lon"], history["v10"][step_idx], {"units": "m/s", "long_name": "10m V-Wind"}),
                "msl": (["lat", "lon"], history["msl"][step_idx] / 100.0, {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
                "tp_1h": (["lat", "lon"], history["tp"][step_idx], {"units": "mm", "long_name": "1-Hour Precipitation Accumulation"}),
                "orog": (["lat", "lon"], orog, {"units": "m", "long_name": "Terrain Elevation"}),
                "lsm": (["lat", "lon"], lsm, {"units": "fraction", "long_name": "Land-Sea Mask"}),
            },
            coords={"lat": lats, "lon": lons},
            attrs={
                "forecast_hour": f"T+{step_idx:02d}h",
                "valid_time_utc": vtime.strftime("%Y-%m-%d %H:%MZ"),
                "institution": "US Air Force Weather / Standalone Regional AI Wx Engine"
            }
        )
        ds_hourly.to_netcdf(hourly_file_path)

    # 2. Export Consolidated Master NetCDF4 Cube
    master_nc_path = os.path.join(output_dir, "forecast_72hr_surface_master.nc")
    ds_master = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], history["t2m"] - 273.15, {"units": "degC", "long_name": "2m Temperature"}),
            "d2m": (["time", "lat", "lon"], history["d2m"] - 273.15, {"units": "degC", "long_name": "2m Dewpoint"}),
            "u10": (["time", "lat", "lon"], history["u10"], {"units": "m/s", "long_name": "10m U-Wind"}),
            "v10": (["time", "lat", "lon"], history["v10"], {"units": "m/s", "long_name": "10m V-Wind"}),
            "msl": (["time", "lat", "lon"], history["msl"] / 100.0, {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
            "tp_1h": (["time", "lat", "lon"], history["tp"], {"units": "mm", "long_name": "1-Hour Precipitation Accumulation"}),
            "orog": (["lat", "lon"], orog, {"units": "m", "long_name": "Terrain Elevation"}),
            "lsm": (["lat", "lon"], lsm, {"units": "fraction", "long_name": "Land-Sea Mask"}),
        },
        coords={"time": valid_times, "lat": lats, "lon": lons}
    )
    ds_master.to_netcdf(master_nc_path)
    print(f"[EXPORT] Saved Master 73-Hour NetCDF4 Cube: {master_nc_path}")

    # 3. Export Center-Point Hourly Meteogram CSV
    mid_lat_idx, mid_lon_idx = len(lats) // 2, len(lons) // 2
    df_rows = []
    for t_idx, vtime in enumerate(valid_times):
        u = history["u10"][t_idx, mid_lat_idx, mid_lon_idx]
        v = history["v10"][t_idx, mid_lat_idx, mid_lon_idx]
        wspd_kt = np.hypot(u, v) * 1.94384
        wdir = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0

        df_rows.append({
            "Forecast_Hour": f"T+{t_idx:02d}h",
            "Valid_Time_UTC": vtime.strftime("%Y-%m-%d %H:%MZ"),
            "Center_Lat": round(float(lats[mid_lat_idx]), 3),
            "Center_Lon": round(float(lons[mid_lon_idx]), 3),
            "Temp_2m_C": round(float(history["t2m"][t_idx, mid_lat_idx, mid_lon_idx] - 273.15), 1),
            "Dewpoint_2m_C": round(float(history["d2m"][t_idx, mid_lat_idx, mid_lon_idx] - 273.15), 1),
            "Wind_Dir_Deg": int(round(wdir)),
            "Wind_Spd_Kts": round(float(wspd_kt), 1),
            "MSLP_hPa": round(float(history["msl"][t_idx, mid_lat_idx, mid_lon_idx] / 100.0), 1),
            "Precip_1hr_mm": round(float(history["tp"][t_idx, mid_lat_idx, mid_lon_idx]), 2)
        })

    csv_path = os.path.join(output_dir, "aor_center_point_forecast_hourly.csv")
    pd.DataFrame(df_rows).to_csv(csv_path, index=False)
    print(f"[EXPORT] Saved 73-Hour Point Meteogram CSV: {csv_path}")

    # 4. Generate Prognostic Surface Charts
    import matplotlib.pyplot as plt
    lon_g, lat_g = np.meshgrid(lons, lats)
    plot_steps = list(range(plot_interval, num_steps, plot_interval))
    if 24 not in plot_steps: plot_steps.append(24)
    if 48 not in plot_steps: plot_steps.append(48)
    if 72 not in plot_steps: plot_steps.append(72)
    plot_steps = sorted(list(set(plot_steps)))

    print(f"[EXPORT] Rendering 4-panel prognostic charts at intervals of {plot_interval} hours...")
    for step_idx in plot_steps:
        fhr = step_idx
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f"AI Regional Surface Forecast | Valid: {valid_times[step_idx].strftime('%Y-%m-%d %H:%MZ')} (T+{fhr:02d}h)", fontsize=14, fontweight="bold")

        # 2m Temp & Coastline
        c1 = axes[0, 0].contourf(lon_g, lat_g, history["t2m"][step_idx] - 273.15, levels=15, cmap="RdYlBu_r")
        axes[0, 0].contour(lon_g, lat_g, lsm, levels=[0.5], colors="black", linewidths=1.2)
        axes[0, 0].set_title("2m Temperature (°C) & Coastline")
        fig.colorbar(c1, ax=axes[0, 0])

        # MSLP & 10m Winds
        msl_hpa = history["msl"][step_idx] / 100.0
        wspd_kt = np.hypot(history["u10"][step_idx], history["v10"][step_idx]) * 1.94384
        c2 = axes[0, 1].contourf(lon_g, lat_g, wspd_kt, levels=12, cmap="YlOrRd")
        cs = axes[0, 1].contour(lon_g, lat_g, msl_hpa, levels=10, colors="black", linewidths=0.8)
        axes[0, 1].clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
        skip = max(1, len(lats) // 12)
        axes[0, 1].barbs(
            lon_g[::skip, ::skip], lat_g[::skip, ::skip],
            history["u10"][step_idx, ::skip, ::skip] * 1.94384,
            history["v10"][step_idx, ::skip, ::skip] * 1.94384,
            length=5
        )
        axes[0, 1].set_title("MSLP (hPa, Contours) & 10m Wind Speed (kts)")
        fig.colorbar(c2, ax=axes[0, 1])

        # 2m Dewpoint
        c3 = axes[1, 0].contourf(lon_g, lat_g, history["d2m"][step_idx] - 273.15, levels=15, cmap="BrBG")
        axes[1, 0].contour(lon_g, lat_g, lsm, levels=[0.5], colors="black", linewidths=1.2)
        axes[1, 0].set_title("2m Dewpoint Temperature (°C)")
        fig.colorbar(c3, ax=axes[1, 0])

        # 1-Hour Precip & Terrain Orography
        c4 = axes[1, 1].contourf(lon_g, lat_g, history["tp"][step_idx], levels=12, cmap="Blues")
        axes[1, 1].contour(lon_g, lat_g, orog, levels=6, colors="saddlebrown", linewidths=0.7, alpha=0.6)
        axes[1, 1].set_title("1-Hour Precip Accumulation (mm) & Terrain Orography")
        fig.colorbar(c4, ax=axes[1, 1])

        png_path = os.path.join(output_dir, f"surface_prog_T+{fhr:02d}h.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

    print(f"[EXPORT] Hourly deliverables generated successfully in: {output_dir}")
