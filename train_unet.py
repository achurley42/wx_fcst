# train_unet.py
import os
import sys
import glob
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import shift

from ingest_grib2 import load_and_regrid_grib2, generate_synthetic_grid
from model_architecture import HybridComputeManager, WeatherSpatioTemporalUNet

SURFACE_KEYS = ["t2m", "d2m", "u10", "v10", "msl", "tp"]


def retrieve_cds_historical_archive(bbox, start_year, end_year, output_dir):
    """
    Downloads multi-month / multi-year 6-hourly ERA5 sequences for model training.
    """
    try:
        import cdsapi
    except ImportError:
        print("[ERROR] 'cdsapi' library is missing. Install with: pip install cdsapi")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    c = cdsapi.Client()
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]

    print("=" * 70)
    print("  AUTOMATED HISTORICAL ARCHIVE RETRIEVAL (CDS API)")
    print("=" * 70)
    print(f"[CDS] Period           : {start_year} through {end_year}")
    print(f"[CDS] Training Cadence : 6-hourly (00Z, 06Z, 12Z, 18Z)")
    print(f"[CDS] Target Directory : {output_dir}")

    years = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    times = ["00:00", "06:00", "12:00", "18:00"]

    for yr in years:
        for mo in months:
            merged_file = os.path.join(output_dir, f"era5_train_{yr}_{mo}.grib2")
            if os.path.exists(merged_file):
                print(f"[SKIP] Archive {merged_file} already exists on disk.")
                continue

            temp_sfc = merged_file + ".sfc.tmp"
            temp_pl = merged_file + ".pl.tmp"

            print(f"\n[CDS] Fetching Surface & Terrain for {yr}-{mo}...")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "grib",
                    "variable": [
                        "2m_temperature",
                        "2m_dewpoint_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure",
                        "total_precipitation",
                        "land_sea_mask",
                        "geopotential",
                    ],
                    "year": yr,
                    "month": mo,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": times,
                    "area": area,
                },
                temp_sfc
            )

            print(f"[CDS] Fetching Isobaric Levels (850/700/500/250 hPa) for {yr}-{mo}...")
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "format": "grib",
                    "variable": [
                        "geopotential",
                        "temperature",
                        "specific_humidity",
                        "u_component_of_wind",
                        "v_component_of_wind",
                    ],
                    "pressure_level": ["250", "500", "700", "850"],
                    "year": yr,
                    "month": mo,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": times,
                    "area": area,
                },
                temp_pl
            )

            with open(merged_file, "wb") as outfile:
                with open(temp_sfc, "rb") as f1: outfile.write(f1.read())
                with open(temp_pl, "rb") as f2: outfile.write(f2.read())

            if os.path.exists(temp_sfc): os.remove(temp_sfc)
            if os.path.exists(temp_pl): os.remove(temp_pl)
            print(f"[CDS] Saved combined training segment: {merged_file}")

    print("\n[SUCCESS] CDS Historical Archive Download Finished.\n")


class HistoricalGribDataset(Dataset):
    def __init__(self, grib_dir, bbox, target_res, num_synthetic_steps=24):
        super().__init__()
        self.bbox = bbox
        self.target_res = target_res
        self.grib_files = sorted(glob.glob(os.path.join(grib_dir, "*.grib*")) +
                                 glob.glob(os.path.join(grib_dir, "*.grb*")))

        self.frames = []
        self.timestamps = []
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

        if len(self.grib_files) >= 2:
            print(f"[DATASET] Ingesting {len(self.grib_files)} historical GRIB2 files from '{grib_dir}'...")
            for idx, fpath in enumerate(self.grib_files):
                grid_data, self.lats, self.lons = load_and_regrid_grib2(fpath, bbox, target_res)
                self.frames.append(grid_data)
                self.timestamps.append(base_time + timedelta(hours=6 * idx))
        else:
            print(f"[DATASET] Insufficient GRIB files found in '{grib_dir}'. Generating synthetic sequence for validation...")
            base_grid, self.lats, self.lons = generate_synthetic_grid(bbox, target_res)
            for step in range(num_synthetic_steps):
                vtime = base_time + timedelta(hours=6 * step)
                self.timestamps.append(vtime)
                self.frames.append(self._evolve_synthetic_state(base_grid, step, vtime))

        self.nlat = len(self.lats)
        self.nlon = len(self.lons)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.dx_meters = abs(self.lats[1] - self.lats[0]) * 111000.0 if self.nlat > 1 else 11100.0

    def _evolve_synthetic_state(self, base_grid, step, vtime):
        state = {k: v.copy() for k, v in base_grid.items()}
        lon_g, lat_g = np.meshgrid(self.lons, self.lats)
        utc_hr = vtime.hour
        local_solar_hr = (utc_hr + lon_g / 15.0) % 24.0
        diurnal = np.cos((local_solar_hr - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        wave_phase = step * 0.35
        lsm = state["lsm"]

        state["t2m"] = base_grid["t2m"] + (0.8 + 3.2 * lsm) * diurnal + 1.8 * np.sin(lon_g * 0.3 - wave_phase)
        state["d2m"] = np.minimum(base_grid["d2m"] + 0.3 * diurnal + 1.2 * np.cos(lon_g * 0.3 - wave_phase), state["t2m"] - 0.5)
        state["u10"] = base_grid["u10"] + 2.0 * np.sin(lat_g * 0.2 + wave_phase)
        state["v10"] = base_grid["v10"] + 2.0 * np.cos(lon_g * 0.25 - wave_phase)
        state["msl"] = base_grid["msl"] + 450.0 * np.cos(lon_g * 0.2 - wave_phase)
        rh = np.clip(100.0 - 5.0 * (state["t2m"] - state["d2m"]), 5.0, 100.0)
        state["tp"] = np.where(rh > 80.0, ((rh - 80.0) / 20.0) ** 2 * 4.0, 0.0).astype(np.float32)
        return state

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, idx):
        curr_grid = self.frames[idx]
        next_grid = self.frames[idx + 1]
        vtime_next = self.timestamps[idx + 1]

        orog = curr_grid["orog"]
        lsm = curr_grid["lsm"]
        orog_norm = (orog - np.mean(orog)) / (np.std(orog) + 1e-5)

        utc_hr = vtime_next.hour
        local_solar_hr = (utc_hr + self.lon_grid / 15.0) % 24.0
        diurnal_wave = np.cos((local_solar_hr - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        prev_diurnal = np.cos(((utc_hr - 6.0 + self.lon_grid / 15.0) % 24.0 - 14.0) * 2.0 * np.pi / 24.0).astype(np.float32)
        diurnal_delta = diurnal_wave - prev_diurnal

        channels = []
        for k in SURFACE_KEYS:
            arr = curr_grid[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        for k in [key for key in curr_grid.keys() if "_" in key]:
            arr = curr_grid[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        channels.extend([orog_norm, lsm, diurnal_wave, np.sin(np.deg2rad(self.lat_grid)).astype(np.float32)])
        while len(channels) < 30:
            channels.append(np.zeros_like(orog_norm))

        x_tensor = torch.tensor(np.stack(channels[:30], axis=0), dtype=torch.float32)

        u_steer = curr_grid.get("u_850", curr_grid["u10"] * 1.5)
        v_steer = curr_grid.get("v_850", curr_grid["v10"] * 1.5)
        shift_x = float(np.mean(u_steer) * 21600.0 / self.dx_meters) * 0.35
        shift_y = float(np.mean(v_steer) * 21600.0 / self.dx_meters) * 0.35

        prev_t2m = shift(curr_grid["t2m"], shift=(shift_y, shift_x), mode="nearest")
        prev_d2m = shift(curr_grid["d2m"], shift=(shift_y, shift_x), mode="nearest")
        prev_u10 = shift(curr_grid["u10"], shift=(shift_y, shift_x), mode="nearest")
        prev_v10 = shift(curr_grid["v10"], shift=(shift_y, shift_x), mode="nearest")
        prev_msl = shift(curr_grid["msl"], shift=(shift_y, shift_x), mode="nearest")

        diurnal_amp = 0.8 + 3.2 * lsm
        base_t2m = prev_t2m + diurnal_amp * diurnal_delta
        base_d2m = prev_d2m + 0.25 * diurnal_amp * diurnal_delta
        base_u10 = prev_u10 * (1.0 - 0.05 * lsm)
        base_v10 = prev_v10 * (1.0 - 0.05 * lsm)
        base_msl = prev_msl

        target_delta = np.stack([
            (next_grid["t2m"] - base_t2m) / 0.5,
            (next_grid["d2m"] - base_d2m) / 0.3,
            (next_grid["u10"] - base_u10) / 0.4,
            (next_grid["v10"] - base_v10) / 0.4,
            (next_grid["msl"] - base_msl) / 50.0,
            next_grid["tp"]
        ], axis=0).astype(np.float32)

        y_tensor = torch.tensor(target_delta, dtype=torch.float32)
        base_t_d_tensor = torch.tensor(np.stack([base_t2m, base_d2m], axis=0), dtype=torch.float32)
        return x_tensor, y_tensor, base_t_d_tensor


class PhysicsInformedWeatherLoss(nn.Module):
    def __init__(self, lats, device):
        super().__init__()
        lat_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
        lat_weights = lat_weights / np.mean(lat_weights)
        self.lat_weights = torch.tensor(lat_weights, device=device).view(1, 1, -1, 1)
        self.huber = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred_delta, target_delta, base_t_d):
        raw_loss = self.huber(pred_delta, target_delta) * self.lat_weights
        mse_loss = raw_loss.mean()
        pred_t2m = base_t_d[:, 0] + pred_delta[:, 0] * 0.5
        pred_d2m = base_t_d[:, 1] + pred_delta[:, 1] * 0.3
        supersat_violation = torch.relu(pred_d2m - pred_t2m)
        physics_penalty = (supersat_violation ** 2).mean() * 2.5
        return mse_loss + physics_penalty


def train_model(args):
    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }

    # Automated Historical CDS Archive Pull Hook
    if args.use_cds:
        retrieve_cds_historical_archive(bbox, args.start_year, args.end_year, args.grib_dir)

    print("=" * 70)
    print("  HYBRID GPU/CPU REGIONAL AI WEATHER MODEL TRAINING PIPELINE")
    print("=" * 70)

    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)
    device = compute_mgr.device

    dataset = HistoricalGribDataset(args.grib_dir, bbox, args.res)
    n_total = len(dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    batch_size = args.batch_size if device.type == "cuda" else min(args.batch_size, 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WeatherSpatioTemporalUNet(in_channels=30, out_channels=6).to(device)
    if args.resume_weights and os.path.exists(args.resume_weights):
        print(f"[TRAIN] Resuming checkpoint: {args.resume_weights}")
        model.load_state_dict(torch.load(args.resume_weights, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = PhysicsInformedWeatherLoss(dataset.lats, device)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)

    print(f"[TRAIN] Epochs: {args.epochs} | Train: {n_train} | Val: {n_val} | Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for x_batch, y_batch, base_td_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            base_td_batch = base_td_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x_batch)
                loss = criterion(pred, y_batch, base_td_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * x_batch.size(0)

        scheduler.step()
        avg_train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_val, y_val, base_td_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                base_td_val = base_td_val.to(device)
                pred_val = model(x_val)
                v_loss = criterion(pred_val, y_val, base_td_val)
                val_loss_sum += v_loss.item() * x_val.size(0)

        avg_val_loss = val_loss_sum / n_val
        status_tag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_weights)
            status_tag = " --> [SAVED BEST CHECKPOINT]"

        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{status_tag}")

    print("=" * 70)
    print(f"[COMPLETE] Model saved to: {args.save_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Fine-Tune Regional AI Weather U-Net on GRIB2 Archives")
    parser.add_argument("--grib-dir", type=str, default="./historical_grib2", help="Directory of 6-hourly GRIB2 files")
    parser.add_argument("--use-cds", action="store_true", help="Download historical archive automatically via CDS API")
    parser.add_argument("--start-year", type=int, default=2023, help="Start year for CDS historical archive")
    parser.add_argument("--end-year", type=int, default=2023, help="End year for CDS historical archive")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum Latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum Latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum Longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum Longitude")
    parser.add_argument("--res", type=float, default=0.10, help="Target spatial resolution in degrees")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (auto-throttled on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save-weights", type=str, default="./checkpoints/regional_unet_best.pt", help="Path for trained weights")
    parser.add_argument("--resume-weights", type=str, default=None, help="Checkpoint to resume fine-tuning")
    parser.add_argument("--force-cpu", action="store_true", help="Force training on CPU")
    train_model(parser.parse_args())
