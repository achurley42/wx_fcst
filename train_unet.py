# train_unet.py
import os
import sys
import glob
import argparse
import shutil
import tempfile
import weakref
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.ndimage import shift

from calibration_registry import SEASONS, save_calibration, season_for_time
from ingest_grib2 import (
    ERA5_LAND_KEYS,
    UPPER_AIR_KEYS,
    generate_synthetic_grid,
    grib_has_era5_land_fields,
    load_and_regrid_grib2_sequence,
)
from model_architecture import (
    MODEL_INPUT_CHANNELS,
    HybridComputeManager,
    WeatherSpatioTemporalUNet,
    load_compatible_state_dict,
)

SURFACE_KEYS = ["t2m", "d2m", "u10", "v10", "msl", "tp"]
RMSE_THRESHOLDS = {
    "t2m": 3.0,
    "d2m": 3.0,
    "u10": 3.0,
    "v10": 3.0,
    "msl": 5.0,
    "tp": 3.0,
}
GRID_FRAME_KEYS = (
    tuple(SURFACE_KEYS)
    + tuple(UPPER_AIR_KEYS)
    + ("orog", "lsm")
    + tuple(ERA5_LAND_KEYS)
)


class _MappedGridFrame:
    def __init__(self, mapped_frames, frame_index, offsets, scales):
        self.mapped_frames = mapped_frames
        self.frame_index = frame_index
        self.offsets = offsets
        self.scales = scales

    def __getitem__(self, key):
        channel = GRID_FRAME_KEYS.index(key)
        values = np.asarray(
            self.mapped_frames[self.frame_index, channel], dtype=np.float32
        )
        return values * self.scales[channel] + self.offsets[channel]

    def get(self, key, default=None):
        if key not in GRID_FRAME_KEYS:
            return default
        return self[key]


def _cleanup_frame_cache(mapped_frames, cache_dir):
    for mapped in mapped_frames:
        mapped._mmap.close()
    shutil.rmtree(cache_dir)


def _days_in_month(year, month):
    import calendar

    return [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)]


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
            if os.path.exists(merged_file) and grib_has_era5_land_fields(merged_file):
                print(f"[SKIP] Archive {merged_file} already exists on disk.")
                continue
            if os.path.exists(merged_file):
                print(f"[CDS] Existing archive lacks ERA5-Land fields; refreshing {merged_file}.")

            temp_sfc = merged_file + ".sfc.tmp"
            temp_pl = merged_file + ".pl.tmp"
            temp_land = merged_file + ".land.tmp"

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
                    "day": _days_in_month(int(yr), int(mo)),
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
                    "day": _days_in_month(int(yr), int(mo)),
                    "time": times,
                    "area": area,
                },
                temp_pl
            )

            print(f"[CDS] Fetching ERA5-Land fields for {yr}-{mo}...")
            c.retrieve(
                "reanalysis-era5-land",
                {
                    "product_type": "reanalysis",
                    "format": "grib",
                    "variable": [
                        "skin_temperature",
                        "soil_temperature_level_1",
                        "volumetric_soil_water_layer_1",
                        "snow_cover",
                    ],
                    "year": yr,
                    "month": mo,
                    "day": _days_in_month(int(yr), int(mo)),
                    "time": times,
                    "area": area,
                },
                temp_land
            )

            with open(merged_file, "wb") as outfile:
                for temp_path in (temp_sfc, temp_pl, temp_land):
                    with open(temp_path, "rb") as data_file:
                        outfile.write(data_file.read())

            for temp_path in (temp_sfc, temp_pl, temp_land):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            print(f"[CDS] Saved combined training segment: {merged_file}")

    print("\n[SUCCESS] CDS Historical Archive Download Finished.\n")


class HistoricalGribDataset(Dataset):
    def __init__(self, grib_dir, bbox, target_res, num_synthetic_steps=24):
        super().__init__()
        self.bbox = bbox
        self.target_res = target_res
        self.grib_files = sorted({
            path
            for path in (
                glob.glob(os.path.join(grib_dir, "*.grib*"))
                + glob.glob(os.path.join(grib_dir, "*.grb*"))
            )
            if not os.path.basename(path).startswith(
                ("era5_sfc_", "era5_pl_", "era5_land_")
            )
            and not path.endswith((".sfc.tmp", ".pl.tmp", ".land.tmp"))
        })
        self.frames = []
        self.timestamps = []
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        cache_dir = tempfile.mkdtemp(prefix="wx_fcst_training_frames-")
        mapped_frames = []
        weakref.finalize(self, _cleanup_frame_cache, mapped_frames, cache_dir)

        if self.grib_files:
            print(f"[DATASET] Ingesting historical GRIB2 time series from '{grib_dir}'...")
            records = []
            for file_index, fpath in enumerate(self.grib_files):
                file_frames, self.lats, self.lons = load_and_regrid_grib2_sequence(
                    fpath, bbox, target_res
                )
                file_frames = [
                    (timestamp, grid_data)
                    for timestamp, grid_data in file_frames
                    if timestamp is not None
                ]
                if not file_frames:
                    continue
                offsets = np.empty(len(GRID_FRAME_KEYS), dtype=np.float32)
                scales = np.empty(len(GRID_FRAME_KEYS), dtype=np.float32)
                for channel_index, key in enumerate(GRID_FRAME_KEYS):
                    channel_min = min(
                        float(np.min(grid_data[key]))
                        for _, grid_data in file_frames
                    )
                    channel_max = max(
                        float(np.max(grid_data[key]))
                        for _, grid_data in file_frames
                    )
                    offsets[channel_index] = (channel_min + channel_max) / 2.0
                    scales[channel_index] = max(
                        1.0,
                        max(
                            abs(channel_min - offsets[channel_index]),
                            abs(channel_max - offsets[channel_index]),
                        ) / 60000.0,
                    )
                cache_path = os.path.join(cache_dir, f"frames_{file_index:04d}.npy")
                mapped = np.lib.format.open_memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        len(file_frames),
                        len(GRID_FRAME_KEYS),
                        len(self.lats),
                        len(self.lons),
                    ),
                )
                for frame_index, (timestamp, grid_data) in enumerate(file_frames):
                    for channel_index, key in enumerate(GRID_FRAME_KEYS):
                        mapped[frame_index, channel_index] = (
                            grid_data[key] - offsets[channel_index]
                        ) / scales[channel_index]
                    records.append(
                        (
                            timestamp,
                            _MappedGridFrame(mapped, frame_index, offsets, scales),
                        )
                    )
                mapped.flush()
                mapped_frames.append(mapped)
                del file_frames
            records.sort(key=lambda record: record[0])
            self.timestamps = [
                timestamp.replace(tzinfo=timezone.utc) for timestamp, _ in records
            ]
            self.frames = [grid_data for _, grid_data in records]

        if len(self.frames) < 2:
            print(f"[DATASET] Insufficient GRIB files found in '{grib_dir}'. Generating synthetic sequence for validation...")
            base_grid, self.lats, self.lons = generate_synthetic_grid(bbox, target_res)
            self.frames = []
            self.timestamps = []
            for step in range(num_synthetic_steps):
                vtime = base_time + timedelta(hours=6 * step)
                self.timestamps.append(vtime)
                self.frames.append(self._evolve_synthetic_state(base_grid, step, vtime))

        self.nlat = len(self.lats)
        self.nlon = len(self.lons)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.dx_meters = abs(self.lats[1] - self.lats[0]) * 111000.0 if self.nlat > 1 else 11100.0
        self.sample_frame_indices = [
            idx
            for idx in range(len(self.frames) - 1)
            if self.timestamps[idx + 1] - self.timestamps[idx] == timedelta(hours=6)
        ]
        self.frame_index_by_time = {
            timestamp: idx for idx, timestamp in enumerate(self.timestamps)
        }

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
        return len(self.sample_frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.sample_frame_indices[idx]
        curr_grid = self.frames[frame_idx]
        next_grid = self.frames[frame_idx + 1]
        vtime_next = self.timestamps[frame_idx + 1]

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
        for k in UPPER_AIR_KEYS:
            arr = curr_grid[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))
        channels.extend([orog_norm, lsm, diurnal_wave, np.sin(np.deg2rad(self.lat_grid)).astype(np.float32)])
        for k in ERA5_LAND_KEYS:
            arr = curr_grid[k]
            channels.append((arr - np.mean(arr)) / (np.std(arr) + 1e-5))

        if len(channels) != MODEL_INPUT_CHANNELS:
            raise ValueError(
                f"Expected {MODEL_INPUT_CHANNELS} model input channels, got {len(channels)}"
            )
        x_tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)

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
            next_grid["tp"] * 1000.0
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


def _sample_target_time(dataset, sample_index):
    frame_index = dataset.sample_frame_indices[sample_index]
    return dataset.timestamps[frame_index + 1]


def _select_rollout_starts(dataset, season, latitude, latest_year):
    candidates = []
    for frame_index, start_time in enumerate(dataset.timestamps):
        if start_time.year != latest_year or season_for_time(start_time, latitude) != season:
            continue
        lead_times = [
            start_time + timedelta(hours=lead)
            for lead in range(6, 73, 6)
        ]
        if all(timestamp in dataset.frame_index_by_time for timestamp in lead_times):
            candidates.append(frame_index)

    return candidates


def _split_rollout_starts(candidates, timestamps, validation_fraction=0.7):
    split_index = max(1, int(len(candidates) * validation_fraction))
    if split_index >= len(candidates):
        return candidates, []
    first_test_start = candidates[split_index]
    validation_candidates = [
        candidate for candidate in candidates[:split_index]
        if timestamps[candidate] + timedelta(hours=72) < timestamps[first_test_start]
    ]
    return validation_candidates, candidates[split_index:]


def _spread_rollout_starts(candidates, max_runs):
    if not candidates:
        return []
    positions = np.linspace(
        0, len(candidates) - 1, num=min(max_runs, len(candidates))
    ).round().astype(int)
    return [candidates[position] for position in positions]


def _evaluate_rollouts(model, dataset, start_indices, compute_mgr, thresholds):
    from forecast_engine import run_hourly_72hr_forecast

    totals = {
        lead: {
            key: {"squared_error": 0.0, "error": 0.0, "count": 0}
            for key in thresholds
        }
        for lead in range(6, 73, 6)
    }
    model.eval()

    for start_index in start_indices:
        start_grid = dataset.frames[start_index]
        start_time = dataset.timestamps[start_index]
        forecast, _ = run_hourly_72hr_forecast(
            grid_data=start_grid,
            lats=dataset.lats,
            lons=dataset.lons,
            model=model,
            compute_mgr=compute_mgr,
            start_time_utc=start_time,
        )
        for lead in range(6, 73, 6):
            valid_time = start_time + timedelta(hours=lead)
            truth = dataset.frames[dataset.frame_index_by_time[valid_time]]
            prediction_index = lead
            for key in thresholds:
                predicted = forecast[key][prediction_index]
                observed = truth[key]
                if key == "tp":
                    observed = observed * 1000.0
                elif key == "msl":
                    predicted = predicted / 100.0
                    observed = observed / 100.0
                difference = np.asarray(predicted, dtype=np.float64) - np.asarray(
                    observed, dtype=np.float64
                )
                finite = np.isfinite(difference)
                totals[lead][key]["squared_error"] += float(np.sum(difference[finite] ** 2))
                totals[lead][key]["error"] += float(np.sum(difference[finite]))
                totals[lead][key]["count"] += int(np.count_nonzero(finite))

    metrics = {}
    accepted = bool(start_indices)
    for lead, variables in totals.items():
        metrics[str(lead)] = {}
        for key, sums in variables.items():
            if sums["count"] == 0:
                rmse = bias = float("inf")
                accepted = False
            else:
                rmse = (sums["squared_error"] / sums["count"]) ** 0.5
                bias = sums["error"] / sums["count"]
                if rmse > thresholds[key]:
                    accepted = False
            metrics[str(lead)][key] = {"rmse": rmse, "bias": bias}
            print(
                f"[VERIFY] T+{lead:02d}h {key}: RMSE={rmse:.3f} "
                f"(limit {thresholds[key]:.3f}), bias={bias:.3f}"
            )
    return metrics, accepted


def _train_calibration_round(model, train_loader, optimizer, criterion, device, epochs):
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_sum = 0.0
    sample_count = 0

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch, base_td_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            base_td_batch = base_td_batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(x_batch)
                loss = criterion(prediction, y_batch, base_td_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * x_batch.size(0)
            sample_count += x_batch.size(0)

    return loss_sum / max(1, sample_count)


def train_model(args):
    if args.max_calibration_rounds < 1:
        raise ValueError("--max-calibration-rounds must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.validation_runs < 1 or args.test_runs < 1:
        raise ValueError("--validation-runs and --test-runs must be at least 1")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (
            args.rmse_temperature_k,
            args.rmse_wind_ms,
            args.rmse_mslp_hpa,
            args.rmse_precip_mm,
        )
    ):
        raise ValueError("All RMSE thresholds must be finite positive values.")

    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }

    if args.use_cds:
        retrieve_cds_historical_archive(bbox, args.start_year, args.end_year, args.grib_dir)

    print("=" * 70)
    print("  SEASONAL REGIONAL FORECAST CALIBRATION")
    print("=" * 70)

    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)
    device = compute_mgr.device
    dataset = HistoricalGribDataset(args.grib_dir, bbox, args.res)
    if not len(dataset):
        raise ValueError("No consecutive six-hour historical frames are available for training.")

    latitude = (args.lat_min + args.lat_max) / 2.0
    latest_year = max(timestamp.year for timestamp in dataset.timestamps)
    batch_size = args.batch_size if device.type == "cuda" else min(args.batch_size, 2)
    criterion = PhysicsInformedWeatherLoss(dataset.lats, device)
    thresholds = {
        "t2m": args.rmse_temperature_k,
        "d2m": args.rmse_temperature_k,
        "u10": args.rmse_wind_ms,
        "v10": args.rmse_wind_ms,
        "msl": args.rmse_mslp_hpa,
        "tp": args.rmse_precip_mm,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)

    failures = []
    for season in SEASONS:
        train_indices = [
            index for index in range(len(dataset))
            if _sample_target_time(dataset, index).year < latest_year
            and season_for_time(_sample_target_time(dataset, index), latitude) == season
        ]
        rollout_candidates = _select_rollout_starts(
            dataset, season, latitude, latest_year
        )
        validation_pool, test_pool = _split_rollout_starts(
            rollout_candidates, dataset.timestamps
        )
        validation_starts = _spread_rollout_starts(
            validation_pool, args.validation_runs
        )
        test_starts = _spread_rollout_starts(test_pool, args.test_runs)
        if not train_indices or not validation_starts or not test_starts:
            print(
                f"[WARN] Skipping {season}: need training samples before {latest_year} "
                "plus distinct calibration-validation and final-test 72-hour cases "
                "in that year."
            )
            failures.append(season)
            continue

        print(
            f"[CALIBRATE] {season}: {len(train_indices)} training transitions, "
            f"{len(validation_starts)} held-out rollouts from {latest_year}."
        )
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
        )
        model = WeatherSpatioTemporalUNet(
            in_channels=MODEL_INPUT_CHANNELS, out_channels=6
        ).to(device)
        if args.resume_weights and os.path.exists(args.resume_weights):
            print(f"[TRAIN] Initializing from checkpoint: {args.resume_weights}")
            load_compatible_state_dict(
                model, torch.load(args.resume_weights, map_location=device)
            )

        learning_rate = args.lr
        accepted = False
        validation_metrics = {}
        test_metrics = {}
        for round_number in range(1, args.max_calibration_rounds + 1):
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-4
            )
            train_loss = _train_calibration_round(
                model, train_loader, optimizer, criterion, device, args.epochs
            )
            print(
                f"[CALIBRATE] {season} round {round_number}/"
                f"{args.max_calibration_rounds}: training loss={train_loss:.5f}"
            )
            validation_metrics, validation_passed = _evaluate_rollouts(
                model, dataset, validation_starts, compute_mgr, thresholds
            )
            if validation_passed:
                print(f"[TEST] Running independent final holdout for {season}.")
                test_metrics, accepted = _evaluate_rollouts(
                    model, dataset, test_starts, compute_mgr, thresholds
                )
                if not accepted:
                    print(
                        f"[REJECTED] {season} passed calibration validation but failed "
                        "the independent final holdout; no weights were registered."
                    )
                    break

                base, extension = os.path.splitext(args.save_weights)
                weights_path = os.path.abspath(f"{base}_{season}{extension or '.pt'}")
                torch.save(model.state_dict(), weights_path)
                save_calibration(
                    args.calibration_registry,
                    {
                        "season": season,
                        "bbox": bbox,
                        "weights_path": weights_path,
                        "validation_year": latest_year,
                        "history_years": sorted({time.year for time in dataset.timestamps}),
                        "validation_rollouts": len(validation_starts),
                        "test_rollouts": len(test_starts),
                        "rmse_thresholds": thresholds,
                        "validation_metrics": validation_metrics,
                        "test_metrics": test_metrics,
                        "input_channels": MODEL_INPUT_CHANNELS,
                    },
                )
                print(
                    f"[ACCEPTED] {season} weights saved and registered: {weights_path}"
                )
                break
            learning_rate *= 0.5
            print(
                f"[CALIBRATE] {season} missed at least one lead/variable threshold; "
                f"reducing learning rate to {learning_rate:.3g} and retraining."
            )

        if not accepted:
            failures.append(season)
            print(
                f"[REJECTED] {season} did not pass all RMSE checks; "
                "no weights were registered."
            )

    if failures:
        raise RuntimeError(
            "Seasonal calibration incomplete for: "
            + ", ".join(failures)
            + ". No unverified weights were added for those seasons."
        )
    print(f"[COMPLETE] Accepted calibrations are recorded in {args.calibration_registry}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Fine-Tune Regional AI Weather U-Net on GRIB2 Archives")
    parser.add_argument("--grib-dir", type=str, default="./historical_grib2", help="Directory of 6-hourly GRIB2 files")
    parser.add_argument("--use-cds", action="store_true", help="Download historical archive automatically via CDS API")
    parser.add_argument(
        "--start-year",
        type=int,
        default=datetime.now(timezone.utc).year - 2,
        help="First year of the default two complete years of historical data",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now(timezone.utc).year - 1,
        help="Last year of the default two complete years of historical data",
    )
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum Latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum Latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum Longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum Longitude")
    parser.add_argument("--res", type=float, default=0.10, help="Target spatial resolution in degrees")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument(
        "--max-calibration-rounds",
        type=int,
        default=5,
        help="Maximum train-and-verify rounds per season",
    )
    parser.add_argument(
        "--validation-runs",
        type=int,
        default=4,
        help="Maximum independent 72-hour validation rollouts per season",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=2,
        help="Maximum separate final 72-hour test rollouts per season",
    )
    parser.add_argument(
        "--rmse-temperature-k",
        type=float,
        default=RMSE_THRESHOLDS["t2m"],
        help="Maximum T+06h...T+72h temperature/dewpoint RMSE in kelvin",
    )
    parser.add_argument(
        "--rmse-wind-ms",
        type=float,
        default=RMSE_THRESHOLDS["u10"],
        help="Maximum 10 m wind-component RMSE in m/s",
    )
    parser.add_argument(
        "--rmse-mslp-hpa",
        type=float,
        default=RMSE_THRESHOLDS["msl"],
        help="Maximum mean sea-level pressure RMSE in hPa",
    )
    parser.add_argument(
        "--rmse-precip-mm",
        type=float,
        default=RMSE_THRESHOLDS["tp"],
        help="Maximum 1-hour precipitation RMSE in mm at each six-hour checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (auto-throttled on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save-weights", type=str, default="./checkpoints/regional_unet_best.pt", help="Path for trained weights")
    parser.add_argument(
        "--calibration-registry",
        type=str,
        default="./checkpoints/calibration_registry.json",
        help="JSON registry for accepted regional and seasonal weights",
    )
    parser.add_argument("--resume-weights", type=str, default=None, help="Checkpoint to resume fine-tuning")
    parser.add_argument("--force-cpu", action="store_true", help="Force training on CPU")
    train_model(parser.parse_args())
