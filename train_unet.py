# train_unet.py
import os
import sys
import glob
import hashlib
import json
import argparse
import tempfile
import weakref
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.ndimage import shift

from calibration_registry import save_calibration
from ingest_grib2 import (
    ERA5_LAND_KEYS,
    UPPER_AIR_KEYS,
    generate_synthetic_grid,
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
REGRID_CACHE_VERSION = 1


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


def _close_mapped_frames(mapped_frames):
    for mapped in mapped_frames:
        mapped._mmap.close()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(stream.isatty() for stream in self.streams if hasattr(stream, "isatty"))


def _configure_training_log(log_path=None):
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = os.path.join("training_logs", f"train_{timestamp}.log")
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[LOG] Writing console output to {log_path}", flush=True)


def _regrid_cache_identity(grib_path, bbox, target_res):
    stat = os.stat(grib_path)
    source = {
        "path": os.path.realpath(grib_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    identity = {
        "cache_version": REGRID_CACHE_VERSION,
        "source": source,
        "bbox": {key: float(value) for key, value in bbox.items()},
        "target_res": float(target_res),
        "frame_keys": list(GRID_FRAME_KEYS),
    }
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["key"] = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    return identity


def _load_regridded_cache(cache_path, identity, mapped_frames):
    metadata_path = os.path.join(cache_path, "metadata.json")
    frames_path = os.path.join(cache_path, "frames.npy")
    if not os.path.isfile(metadata_path) or not os.path.isfile(frames_path):
        return None

    mapped = None
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("identity") != identity:
            print(f"[CACHE] Metadata mismatch; rebuilding {cache_path}.", flush=True)
            return None
        mapped = np.load(frames_path, mmap_mode="r", allow_pickle=False)
        lats = np.asarray(metadata["lats"], dtype=np.float64)
        lons = np.asarray(metadata["lons"], dtype=np.float64)
        offsets = np.asarray(metadata["offsets"], dtype=np.float32)
        scales = np.asarray(metadata["scales"], dtype=np.float32)
        timestamps = [
            datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            for value in metadata["timestamps"]
        ]
        expected_shape = (
            len(timestamps),
            len(GRID_FRAME_KEYS),
            len(lats),
            len(lons),
        )
        if (
            mapped.dtype != np.float16
            or mapped.shape != expected_shape
            or offsets.shape != (len(GRID_FRAME_KEYS),)
            or scales.shape != (len(GRID_FRAME_KEYS),)
        ):
            mapped._mmap.close()
            print(f"[CACHE] Invalid cached frame shape; rebuilding {cache_path}.", flush=True)
            return None
        mapped_frames.append(mapped)
        records = [
            (
                timestamp,
                _MappedGridFrame(mapped, frame_index, offsets, scales),
            )
            for frame_index, timestamp in enumerate(timestamps)
        ]
        print(
            f"[CACHE] Reusing {len(records)} regridded frames from {cache_path}.",
            flush=True,
        )
        return records, lats, lons
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if mapped is not None and all(mapped is not item for item in mapped_frames):
            mapped._mmap.close()
        print(
            f"[CACHE] Could not load regridded cache {cache_path}: {exc}; "
            "rebuilding from GRIB.",
            flush=True,
        )
        return None


def _save_regridded_cache(
    cache_path, identity, frames, lats, lons, offsets, scales, mapped_frames
):
    os.makedirs(cache_path, exist_ok=True)
    metadata_path = os.path.join(cache_path, "metadata.json")
    frames_path = os.path.join(cache_path, "frames.npy")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    temp_frames_path = os.path.join(
        cache_path, f"frames.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        mapped = np.lib.format.open_memmap(
            temp_frames_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(frames),
                len(GRID_FRAME_KEYS),
                len(lats),
                len(lons),
            ),
        )
        timestamps = []
        try:
            for frame_index, (timestamp, grid_data) in enumerate(frames):
                timestamps.append(timestamp.isoformat())
                for channel_index, key in enumerate(GRID_FRAME_KEYS):
                    mapped[frame_index, channel_index] = (
                        grid_data[key] - offsets[channel_index]
                    ) / scales[channel_index]
            mapped.flush()
        finally:
            mapped._mmap.close()
    except Exception:
        if os.path.exists(temp_frames_path):
            os.remove(temp_frames_path)
        raise
    os.replace(temp_frames_path, frames_path)

    metadata = {
        "identity": identity,
        "lats": np.asarray(lats).tolist(),
        "lons": np.asarray(lons).tolist(),
        "offsets": offsets.tolist(),
        "scales": scales.tolist(),
        "timestamps": timestamps,
    }
    fd, temp_metadata_path = tempfile.mkstemp(
        prefix=".metadata-", suffix=".json", dir=cache_path
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, sort_keys=True)
            metadata_file.write("\n")
        os.replace(temp_metadata_path, metadata_path)
    except Exception:
        if os.path.exists(temp_metadata_path):
            os.remove(temp_metadata_path)
        raise

    print(f"[CACHE] Saved {len(frames)} regridded frames to {cache_path}.", flush=True)
    cached = _load_regridded_cache(cache_path, identity, mapped_frames)
    if cached is None:
        raise RuntimeError(f"Could not reopen newly written regrid cache: {cache_path}")
    return cached


def _days_in_month(year, month):
    import calendar

    return [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def _format_bbox_bound(value):
    return format(float(value), ".10g")


def _historical_archive_path(output_dir, bbox, year, month):
    lat_bounds = (
        f"{_format_bbox_bound(bbox['lat_min'])}_"
        f"{_format_bbox_bound(bbox['lat_max'])}"
    )
    lon_bounds = (
        f"{_format_bbox_bound(bbox['lon_min'])}_"
        f"{_format_bbox_bound(bbox['lon_max'])}"
    )
    filename = (
        f"era5_train_lat{lat_bounds}_lon{lon_bounds}_"
        f"{year:04d}_{month:02d}.grib2"
    )
    return os.path.join(output_dir, filename)


def _existing_historical_archives(output_dir, bbox, start_year, end_year):
    existing = {}
    total = (end_year - start_year + 1) * 12
    print(
        f"[CDS] Checking for {total} monthly archives "
        f"({start_year}-{end_year}) matching "
        f"lat {bbox['lat_min']}..{bbox['lat_max']}, "
        f"lon {bbox['lon_min']}..{bbox['lon_max']} in '{output_dir}'..."
    )
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            archive_path = _historical_archive_path(
                output_dir, bbox, year, month
            )
            if not os.path.isfile(archive_path):
                print(f"[CDS] NOT FOUND: {archive_path}; it will be downloaded.")
                existing[(year, month)] = False
                continue
            if os.path.getsize(archive_path) == 0:
                print(f"[CDS] EMPTY: {archive_path}; it will be downloaded again.")
                existing[(year, month)] = False
                continue
            archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
            print(
                f"[CDS] FOUND ({archive_size_mb:.1f} MB): {archive_path}; "
                "reusing without scanning GRIB contents.",
                flush=True,
            )
            existing[(year, month)] = True

    reusable = sum(existing.values())
    print(f"[CDS] Archive check complete: {reusable}/{total} monthly archives reusable.")
    return existing


def _list_historical_grib_files(grib_dir, expected_archives=None):
    if expected_archives is not None:
        missing_archives = [
            path
            for path in expected_archives
            if not os.path.isfile(path) or os.path.getsize(path) == 0
        ]
        if missing_archives:
            raise ValueError(
                "Expected nonempty, bounds-matched CDS archives are missing "
                "after retrieval: " + ", ".join(missing_archives)
            )
        return list(expected_archives)

    grib_patterns = ("*.grib", "*.grib2", "*.grb", "*.grb2")
    return sorted({
        path
        for pattern in grib_patterns
        for path in glob.glob(os.path.join(grib_dir, pattern))
        if not os.path.basename(path).startswith(
            ("era5_sfc_", "era5_pl_", "era5_land_")
        )
        and not path.endswith((".sfc.tmp", ".pl.tmp", ".land.tmp"))
    })


def retrieve_cds_historical_archive(bbox, start_year, end_year, output_dir):
    """
    Downloads multi-month / multi-year 6-hourly ERA5 sequences for model training.
    """
    if start_year > end_year:
        raise ValueError("Historical archive start year must not exceed end year.")

    os.makedirs(output_dir, exist_ok=True)
    existing_archives = _existing_historical_archives(
        output_dir, bbox, start_year, end_year
    )
    if all(existing_archives.values()):
        print(
            f"[CDS] All nonempty, bounds-matched archives for "
            f"{start_year}-{end_year} are present; skipping downloads."
        )
        return
    missing_count = len(existing_archives) - sum(existing_archives.values())
    print(
        f"[CDS] {missing_count} monthly archive(s) are missing or incomplete; "
        "downloading those months before calibration."
    )

    try:
        import cdsapi
    except ImportError:
        print("[ERROR] 'cdsapi' library is missing. Install with: pip install cdsapi")
        sys.exit(1)

    c = cdsapi.Client()
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]

    print("=" * 70)
    print("  AUTOMATED HISTORICAL ARCHIVE RETRIEVAL (CDS API)")
    print("=" * 70)
    print(f"[CDS] Period           : {start_year} through {end_year}")
    print(f"[CDS] Training Cadence : 6-hourly (00Z, 06Z, 12Z, 18Z)")
    print(f"[CDS] Target Directory : {output_dir}")

    years = range(start_year, end_year + 1)
    months = [f"{m:02d}" for m in range(1, 13)]
    times = ["00:00", "06:00", "12:00", "18:00"]

    for year in years:
        yr = str(year)
        for mo in months:
            month = int(mo)
            merged_file = _historical_archive_path(
                output_dir, bbox, year, month
            )
            if existing_archives[(year, month)]:
                print(f"[SKIP] Archive {merged_file} already exists on disk.")
                continue
            if os.path.exists(merged_file):
                print(
                    f"[CDS] Existing bounds-matched archive is empty; "
                    f"refreshing {merged_file}."
                )

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
    def __init__(
        self,
        grib_dir,
        bbox,
        target_res,
        num_synthetic_steps=24,
        expected_archives=None,
        cache_dir=None,
    ):
        super().__init__()
        self.bbox = bbox
        self.target_res = target_res
        self.grib_files = _list_historical_grib_files(
            grib_dir, expected_archives
        )
        self.frames = []
        self.timestamps = []
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        mapped_frames = []
        weakref.finalize(self, _close_mapped_frames, mapped_frames)

        if self.grib_files:
            cache_root = os.path.abspath(
                os.path.expanduser(
                    cache_dir
                    or os.path.join("~", ".cache", "wx_fcst", "regrid")
                )
            )
            os.makedirs(cache_root, exist_ok=True)
            print(f"[DATASET] Ingesting historical GRIB2 time series from '{grib_dir}'...")
            records = []
            for file_index, fpath in enumerate(self.grib_files):
                file_started = time.perf_counter()
                print(
                    f"[DATASET] Processing archive {file_index + 1}/"
                    f"{len(self.grib_files)}: {fpath}"
                )
                identity = _regrid_cache_identity(fpath, bbox, target_res)
                cache_path = os.path.join(cache_root, identity["key"])
                cached = _load_regridded_cache(
                    cache_path, identity, mapped_frames
                )
                if cached is not None:
                    cached_records, self.lats, self.lons = cached
                    records.extend(cached_records)
                    print(
                        f"[DATASET] Loaded cached archive in "
                        f"{time.perf_counter() - file_started:.1f}s."
                    )
                    continue

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
                cached = _save_regridded_cache(
                    cache_path,
                    identity,
                    file_frames,
                    self.lats,
                    self.lons,
                    offsets,
                    scales,
                    mapped_frames,
                )
                cached_records, self.lats, self.lons = cached
                records.extend(cached_records)
                del file_frames
                print(
                    f"[DATASET] Cached {len(records)} cumulative frames after "
                    f"{time.perf_counter() - file_started:.1f}s."
                )
            records.sort(key=lambda record: record[0])
            self.timestamps = [
                timestamp.replace(tzinfo=timezone.utc) for timestamp, _ in records
            ]
            self.frames = [grid_data for _, grid_data in records]

        if len(self.frames) < 2:
            if expected_archives is not None:
                raise ValueError(
                    "The requested CDS archives produced fewer than two "
                    "timestamped ERA5 frames; refusing to train on synthetic data."
                )
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


def _select_rollout_starts(dataset, month, latest_year):
    candidates = []
    for frame_index, start_time in enumerate(dataset.timestamps):
        if start_time.year != latest_year or start_time.month != month:
            continue
        lead_times = [
            start_time + timedelta(hours=lead)
            for lead in range(6, 73, 6)
        ]
        if all(timestamp in dataset.frame_index_by_time for timestamp in lead_times):
            candidates.append(frame_index)

    return candidates


def _validation_score(metrics, thresholds):
    normalized_rmse = [
        values[key]["rmse"] / thresholds[key]
        for values in metrics.values()
        for key in thresholds
    ]
    if not normalized_rmse or not all(np.isfinite(value) for value in normalized_rmse):
        return float("inf")
    return float(np.mean(normalized_rmse))


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


def _round_epoch_budgets(total_epochs, max_rounds):
    round_count = min(total_epochs, max_rounds)
    base_epochs, extra_epochs = divmod(total_epochs, round_count)
    return [
        base_epochs + (1 if index < extra_epochs else 0)
        for index in range(round_count)
    ]


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
    round_loss_sum = 0.0
    round_sample_count = 0
    batches_per_epoch = len(train_loader)
    log_interval = max(1, batches_per_epoch // 10)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        model.train()
        for batch_index, (x_batch, y_batch, base_td_batch) in enumerate(train_loader, 1):
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
            batch_loss = loss.item() * x_batch.size(0)
            epoch_loss_sum += batch_loss
            epoch_sample_count += x_batch.size(0)
            if batch_index % log_interval == 0 or batch_index == batches_per_epoch:
                print(
                    f"[TRAIN] Epoch {epoch}/{epochs}: batch "
                    f"{batch_index}/{batches_per_epoch}, "
                    f"loss={loss.item():.5f}"
                )

        epoch_loss = epoch_loss_sum / max(1, epoch_sample_count)
        round_loss_sum += epoch_loss_sum
        round_sample_count += epoch_sample_count
        print(
            f"[TRAIN] Epoch {epoch}/{epochs} complete: "
            f"mean loss={epoch_loss:.5f}, "
            f"elapsed={time.perf_counter() - epoch_started:.1f}s"
        )

    return round_loss_sum / max(1, round_sample_count)


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
        print(
            f"[DATA] CDS archive check enabled for {args.start_year}-{args.end_year}; "
            "calibration will start after required monthly files are available."
        )
        retrieve_cds_historical_archive(bbox, args.start_year, args.end_year, args.grib_dir)
    else:
        print(
            f"[DATA] CDS archive check/download is disabled; using local GRIB files "
            f"from '{args.grib_dir}'. Pass --use-cds to check and download missing "
            "monthly archives before calibration."
        )

    print("=" * 70)
    print("  MONTHLY REGIONAL FORECAST CALIBRATION")
    print("=" * 70)

    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)
    device = compute_mgr.device
    expected_archives = None
    if args.use_cds:
        expected_archives = [
            _historical_archive_path(args.grib_dir, bbox, year, month)
            for year in range(args.start_year, args.end_year + 1)
            for month in range(1, 13)
        ]
    dataset = HistoricalGribDataset(
        args.grib_dir,
        bbox,
        args.res,
        expected_archives=expected_archives,
        cache_dir=args.cache_dir,
    )
    if not len(dataset):
        raise ValueError("No consecutive six-hour historical frames are available for training.")

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
    fallback_months = []
    for month in range(1, 13):
        month_name = datetime(2000, month, 1).strftime("%B")
        train_indices = [
            index for index in range(len(dataset))
            if _sample_target_time(dataset, index).year < latest_year
            and _sample_target_time(dataset, index).month == month
        ]
        rollout_candidates = _select_rollout_starts(
            dataset, month, latest_year
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
                f"[WARN] Skipping {month_name}: need training samples before {latest_year} "
                "plus distinct calibration-validation and final-test 72-hour cases "
                "in that year."
            )
            failures.append(month_name)
            continue

        print(
            f"[CALIBRATE] {month_name}: {len(train_indices)} training transitions, "
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
        best_state = None
        best_validation_metrics = {}
        best_validation_score = float("inf")
        best_validation_passed = False
        round_epoch_budgets = _round_epoch_budgets(
            args.epochs, args.max_calibration_rounds
        )
        round_count = len(round_epoch_budgets)
        if round_count < args.max_calibration_rounds:
            print(
                f"[CALIBRATE] Limiting rounds to {round_count}: "
                f"--epochs={args.epochs} allows at least one training epoch per round."
            )
        print(
            f"[CALIBRATE] Training budget: {args.epochs} epochs "
            f"({sum(round_epoch_budgets)} total) for {month_name}, "
            f"distributed across {round_count} rounds."
        )
        for round_number, round_epochs in enumerate(round_epoch_budgets, 1):
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-4
            )
            train_loss = _train_calibration_round(
                model, train_loader, optimizer, criterion, device, round_epochs
            )
            print(
                f"[CALIBRATE] {month_name} round {round_number}/"
                f"{round_count} ({round_epochs} epoch(s)): "
                f"training loss={train_loss:.5f}"
            )
            validation_metrics, validation_passed = _evaluate_rollouts(
                model, dataset, validation_starts, compute_mgr, thresholds
            )
            validation_score = _validation_score(validation_metrics, thresholds)
            better_candidate = (
                best_state is None
                or (validation_passed and not best_validation_passed)
                or (
                    validation_passed == best_validation_passed
                    and validation_score < best_validation_score
                )
            )
            if better_candidate:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_validation_metrics = validation_metrics
                best_validation_score = validation_score
                best_validation_passed = validation_passed
                print(
                    f"[CALIBRATE] {month_name} round {round_number} is the "
                    f"best validation candidate so far "
                    f"(normalized RMSE={validation_score:.4f})."
                )
            if validation_passed:
                break
            learning_rate *= 0.5
            print(
                f"[CALIBRATE] {month_name} missed at least one lead/variable "
                "threshold; "
                f"reducing learning rate to {learning_rate:.3g} and retraining."
            )

        if best_state is None or not np.isfinite(best_validation_score):
            failures.append(month_name)
            print(
                f"[ERROR] {month_name} produced no finite validation candidate; "
                "no weights were registered."
            )
            continue

        model.load_state_dict(best_state)
        print(
            f"[TEST] Evaluating best {month_name} validation candidate on the "
            "independent final holdout."
        )
        test_metrics, test_passed = _evaluate_rollouts(
            model, dataset, test_starts, compute_mgr, thresholds
        )
        accepted = best_validation_passed and test_passed
        status = "accepted" if accepted else "fallback"

        base, extension = os.path.splitext(args.save_weights)
        month_slug = datetime(2000, month, 1).strftime("%b").lower()
        weight_suffix = (
            f"_{month_slug}" if accepted else f"_{month_slug}_fallback"
        )
        weights_path = os.path.abspath(
            f"{base}{weight_suffix}{extension or '.pt'}"
        )
        torch.save(model.state_dict(), weights_path)
        save_calibration(
            args.calibration_registry,
            {
                "month": month,
                "status": status,
                "validation_score": best_validation_score,
                "bbox": bbox,
                "weights_path": weights_path,
                "validation_year": latest_year,
                "history_years": sorted({time.year for time in dataset.timestamps}),
                "validation_rollouts": len(validation_starts),
                "test_rollouts": len(test_starts),
                "rmse_thresholds": thresholds,
                "validation_metrics": best_validation_metrics,
                "test_metrics": test_metrics,
                "input_channels": MODEL_INPUT_CHANNELS,
            },
        )
        if accepted:
            print(
                f"[ACCEPTED] {month_name} weights saved and registered: "
                f"{weights_path}"
            )
        else:
            fallback_months.append(month_name)
            print(
                f"[FALLBACK] {month_name} best validation weights saved and "
                f"registered for use despite missing one or more RMSE limits: "
                f"{weights_path}"
            )

    if failures:
        raise RuntimeError(
            "Monthly calibration incomplete for: "
            + ", ".join(failures)
            + ". No weights were registered for those months."
        )
    if fallback_months:
        print(
            "[COMPLETE] Monthly calibrations saved. These months use best-score "
            "fallback weights that did not pass every threshold: "
            + ", ".join(fallback_months)
        )
    print(f"[COMPLETE] Monthly calibrations are recorded in {args.calibration_registry}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train monthly regional AI weather weights from GRIB2 archives"
    )
    parser.add_argument("--grib-dir", type=str, default="./historical_grib2", help="Directory of 6-hourly GRIB2 files")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join("~", ".cache", "wx_fcst", "regrid"),
        help="Persistent per-archive regridding cache directory",
    )
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epoch budget per monthly calibration",
    )
    parser.add_argument(
        "--max-calibration-rounds",
        type=int,
        default=5,
        help="Maximum train-and-verify rounds per month",
    )
    parser.add_argument(
        "--validation-runs",
        type=int,
        default=4,
        help="Maximum independent 72-hour validation rollouts per month",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=2,
        help="Maximum separate final 72-hour test rollouts per month",
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
        help="JSON registry for regional and monthly weights",
    )
    parser.add_argument("--resume-weights", type=str, default=None, help="Checkpoint to resume fine-tuning")
    parser.add_argument("--force-cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: ./training_logs/train_<timestamp>.log)",
    )
    args = parser.parse_args()
    _configure_training_log(args.log_file)
    train_model(args)
