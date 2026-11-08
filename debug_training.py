import argparse
import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone


_GRIB_WORKER = r"""
import json
import sys
import traceback
import numpy as np
import cfgrib
from ingest_grib2 import (
    ERA5_LAND_VARS,
    REQUIRED_PLEVS,
    SURFACE_VARS,
    UPPER_AIR_VARS,
    _available_valid_times,
    _regrid_datasets_sequence,
)

path, lat_min, lat_max, lon_min, lon_max, resolution, check_regrid = sys.argv[1:]
bbox = {
    "lat_min": float(lat_min),
    "lat_max": float(lat_max),
    "lon_min": float(lon_min),
    "lon_max": float(lon_max),
}
datasets = []
try:
    datasets = cfgrib.open_datasets(
        path, backend_kwargs={"indexpath": ""}
    )
    variables = set().union(*(set(ds.data_vars) for ds in datasets))
    missing_surface = [
        key for key, aliases in SURFACE_VARS.items()
        if not variables.intersection(aliases)
    ]
    missing_land = [
        key for key, aliases in ERA5_LAND_VARS.items()
        if not variables.intersection(aliases)
    ]
    missing_upper = []
    for key, aliases in UPPER_AIR_VARS.items():
        variable = next(
            (ds[alias] for ds in datasets for alias in aliases if alias in ds.data_vars),
            None,
        )
        if variable is None or "isobaricInhPa" not in variable.coords:
            missing_upper.extend(f"{key}_{level}" for level in REQUIRED_PLEVS)
            continue
        levels = set(np.asarray(variable["isobaricInhPa"].values).astype(int).tolist())
        missing_upper.extend(
            f"{key}_{level}" for level in REQUIRED_PLEVS if level not in levels
        )

    sample = next(
        (
            ds for ds in datasets
            if ("latitude" in ds.coords or "lat" in ds.coords)
            and ("longitude" in ds.coords or "lon" in ds.coords)
        ),
        None,
    )
    source_bounds = None
    if sample is not None:
        lat_name = "latitude" if "latitude" in sample.coords else "lat"
        lon_name = "longitude" if "longitude" in sample.coords else "lon"
        latitudes = np.asarray(sample[lat_name].values, dtype=float)
        longitudes = np.asarray(sample[lon_name].values, dtype=float)
        if np.nanmax(longitudes) > 180:
            longitudes = ((longitudes + 180) % 360) - 180
        source_bounds = {
            "lat_min": float(np.nanmin(latitudes)),
            "lat_max": float(np.nanmax(latitudes)),
            "lon_min": float(np.nanmin(longitudes)),
            "lon_max": float(np.nanmax(longitudes)),
        }

    valid_times = _available_valid_times(datasets)
    result = {
        "status": "opened",
        "groups": len(datasets),
        "variables": sorted(variables),
        "missing_surface": missing_surface,
        "missing_land": missing_land,
        "missing_upper": missing_upper,
        "source_bounds": source_bounds,
        "timestamps": [
            str(timestamp.astype("datetime64[ns]"))
            for timestamp in valid_times
            if not np.isnat(timestamp)
        ],
    }
    if check_regrid:
        if not valid_times.size:
            raise ValueError("No valid_time/time coordinate found for regrid check.")
        frames, lats, lons = _regrid_datasets_sequence(
            datasets, bbox, float(resolution), valid_times
        )
        result["regrid"] = {
            "frames": len(frames),
            "grid_shape": [len(lats), len(lons)],
        }
    print("DEBUG_RESULT=" + json.dumps(result), flush=True)
except Exception as exc:
    print(
        "DEBUG_RESULT="
        + json.dumps(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        ),
        flush=True,
    )
finally:
    for dataset in datasets:
        dataset.close()
"""


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return self.streams[0].isatty()


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only preflight for monthly ERA5 training archives. "
            "Reports go to the screen; optionally tee them to a file. "
            "No GRIB index is written."
        )
    )
    parser.add_argument("--grib-dir", default="./historical_grib2")
    parser.add_argument(
        "--cache-dir", default="~/.cache/wx_fcst/regrid"
    )
    parser.add_argument("--use-cds", action="store_true")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=35.0)
    parser.add_argument("--lon-min", type=float, default=-90.0)
    parser.add_argument("--lon-max", type=float, default=-80.0)
    parser.add_argument("--res", type=float, default=0.10)
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=90.0,
        help="Seconds allowed for each uncached GRIB open (default: 90).",
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=3,
        help="Maximum uncached GRIBs to inspect; use 0 for all.",
    )
    parser.add_argument(
        "--check-regrid",
        action="store_true",
        help="Also run the full in-memory regrid check in each inspected child process.",
    )
    parser.add_argument(
        "--output-file",
        help="Also write screen output to this file (overwrites it if it exists).",
    )
    return parser.parse_args()


def _expected_archive_path(output_dir, bbox, year, month):
    def bound(value):
        return format(float(value), ".10g")

    filename = (
        f"era5_train_lat{bound(bbox['lat_min'])}_{bound(bbox['lat_max'])}"
        f"_lon{bound(bbox['lon_min'])}_{bound(bbox['lon_max'])}"
        f"_{year:04d}_{month:02d}.grib2"
    )
    return os.path.join(output_dir, filename)


def _find_local_grib_files(grib_dir):
    extensions = (".grib", ".grib2", ".grb", ".grb2")
    excluded_prefixes = ("era5_sfc_", "era5_pl_", "era5_land_")
    return sorted(
        os.path.join(grib_dir, name)
        for name in os.listdir(grib_dir)
        if name.lower().endswith(extensions)
        and not name.startswith(excluded_prefixes)
        and not name.endswith((".sfc.tmp", ".pl.tmp", ".land.tmp"))
    ) if os.path.isdir(grib_dir) else []


def _check_imports(use_cds):
    modules = ["numpy", "torch", "scipy", "xarray", "cfgrib"]
    if use_cds:
        modules.append("cdsapi")
    available = {}
    print("[ENV] Python:", sys.executable)
    print("[ENV] Version:", sys.version.split()[0])
    for name in modules:
        try:
            module = importlib.import_module(name)
            available[name] = True
            print(
                f"[ENV] {name}: available "
                f"({getattr(module, '__version__', 'version unknown')})"
            )
        except Exception as exc:
            available[name] = False
            print(f"[ENV] {name}: UNAVAILABLE ({type(exc).__name__}: {exc})")
    return available


def _cache_status(path, bbox, resolution, cache_root, training):
    identity = training._regrid_cache_identity(path, bbox, resolution)
    cache_path = os.path.join(cache_root, identity["key"])
    metadata_path = os.path.join(cache_path, "metadata.json")
    frames_path = os.path.join(cache_path, "frames.npy")
    if not os.path.isfile(metadata_path) or not os.path.isfile(frames_path):
        return "MISS", [], cache_path
    mapped_frames = []
    try:
        cached = training._load_regridded_cache(
            cache_path, identity, mapped_frames
        )
        if cached is None:
            return "INVALID", [], cache_path
        records, _, _ = cached
        timestamps = [record[0] for record in records]
        return "HIT", timestamps, cache_path
    except Exception as exc:
        print(f"[CACHE] Could not inspect {cache_path}: {exc}")
        return "INVALID", [], cache_path
    finally:
        training._close_mapped_frames(mapped_frames)


def _run_worker(path, bbox, resolution, timeout, check_regrid):
    command = [
        sys.executable,
        "-c",
        _GRIB_WORKER,
        os.path.abspath(path),
        str(bbox["lat_min"]),
        str(bbox["lat_max"]),
        str(bbox["lon_min"]),
        str(bbox["lon_max"]),
        str(resolution),
        "1" if check_regrid else "0",
    ]
    process = subprocess.Popen(
        command,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    started = time.monotonic()
    interval = min(15.0, timeout)
    while True:
        elapsed = time.monotonic() - started
        remaining = timeout - elapsed
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            return "TIMEOUT", None, stdout, stderr, elapsed
        try:
            stdout, stderr = process.communicate(
                timeout=min(interval, remaining)
            )
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(
                f"  [WAIT] Child inspection still running "
                f"({elapsed:.1f}/{timeout:.1f}s)...",
                flush=True,
            )
        except KeyboardInterrupt:
            process.kill()
            process.communicate()
            raise

    result = None
    for line in stdout.splitlines():
        if line.startswith("DEBUG_RESULT="):
            try:
                result = json.loads(line[len("DEBUG_RESULT="):])
            except json.JSONDecodeError:
                result = None
    status = "OK" if process.returncode == 0 and result else "ERROR"
    return status, result, stdout, stderr, time.monotonic() - started


def _report_time_coverage(timestamps, training, complete):
    unique_times = sorted(set(timestamps))
    if not unique_times:
        print("[DATA] No inspectable timestamps were available for coverage checks.")
        return
    latest_year = max(timestamp.year for timestamp in unique_times)
    time_set = set(unique_times)
    print(
        f"[DATA] {len(unique_times)} unique timestamp(s) inspected; "
        f"latest year is {latest_year}."
        + ("" if complete else " Coverage is partial.")
    )
    incomplete_months = []
    for month in range(1, 13):
        train_count = sum(
            1
            for current, following in zip(unique_times, unique_times[1:])
            if following - current == timedelta(hours=6)
            and following.year < latest_year
            and following.month == month
        )
        rollout_candidates = [
            index
            for index, start in enumerate(unique_times)
            if start.year == latest_year
            and start.month == month
            and all(
                start + timedelta(hours=lead) in time_set
                for lead in range(6, 73, 6)
            )
        ]
        validation, test = training._split_rollout_starts(
            rollout_candidates, unique_times
        )
        print(
            f"[DATA] {datetime(2000, month, 1).strftime('%B')}: "
            f"{train_count} training transition(s), "
            f"{len(validation)} validation candidate(s), "
            f"{len(test)} test candidate(s)"
        )
        if not train_count or not validation or not test:
            incomplete_months.append(datetime(2000, month, 1).strftime("%b"))
    if incomplete_months:
        print(
            "[DATA] Months that may lack a training or holdout split based on "
            "the inspected timestamps: " + ", ".join(incomplete_months)
        )


def main():
    args = _parse_args()
    if not args.output_file:
        return _run_diagnostic(args)

    output_path = os.path.abspath(os.path.expanduser(args.output_file))
    try:
        output_file = open(output_path, "w", encoding="utf-8", buffering=1)
    except OSError as exc:
        print(
            f"[ERROR] Could not open output file {output_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    with output_file:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, output_file)))
            stack.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, output_file)))
            print(f"[OUTPUT] Also writing console output to: {output_path}")
            return _run_diagnostic(args)


def _run_diagnostic(args):
    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }
    print("=" * 72)
    print("TRAINING PREFLIGHT DIAGNOSTIC")
    print("=" * 72)
    print(
        "[CONFIG] Bounds="
        f"{bbox['lat_min']}..{bbox['lat_max']} latitude, "
        f"{bbox['lon_min']}..{bbox['lon_max']} longitude; "
        f"resolution={args.res}°"
    )
    print(f"[CONFIG] GRIB directory: {os.path.abspath(args.grib_dir)}")
    print(f"[CONFIG] Regrid cache: {os.path.expanduser(args.cache_dir)}")

    if (
        not -90 <= args.lat_min < args.lat_max <= 90
        or not -360 <= args.lon_min < args.lon_max <= 360
        or args.res <= 0
        or args.start_year > args.end_year
    ):
        print("[ERROR] Invalid bounds, resolution, or year range.")
        return 2
    if args.open_timeout <= 0 or args.max_open_files < 0:
        print("[ERROR] Timeout must be positive and max-open-files nonnegative.")
        return 2

    imports = _check_imports(args.use_cds)
    missing_modules = [name for name, available in imports.items() if not available]
    if missing_modules:
        print(
            "[STOP] Required/import-requested modules unavailable: "
            + ", ".join(missing_modules)
        )
        return 2

    try:
        import train_unet as training
    except Exception as exc:
        print(f"[STOP] Could not import training pipeline: {type(exc).__name__}: {exc}")
        return 2

    if args.use_cds:
        archives = [
            _expected_archive_path(args.grib_dir, bbox, year, month)
            for year in range(args.start_year, args.end_year + 1)
            for month in range(1, 13)
        ]
    else:
        archives = _find_local_grib_files(args.grib_dir)
    print(f"[DATA] Selected {len(archives)} GRIB archive(s).")
    if not archives:
        print("[ERROR] No GRIB archives matched the selected mode and directory.")
        return 1

    archive_times = []
    cache_misses = []
    cache_root = os.path.abspath(os.path.expanduser(args.cache_dir))
    missing_archives = []
    for path in archives:
        if not os.path.isfile(path):
            print(f"[DATA] MISSING: {path}")
            missing_archives.append(path)
            continue
        size = os.path.getsize(path)
        if size == 0:
            print(f"[DATA] EMPTY: {path}")
            missing_archives.append(path)
            continue
        print(f"[DATA] FOUND: {path} ({size / (1024 * 1024):.1f} MiB)")
        status, timestamps, cache_path = _cache_status(
            path, bbox, args.res, cache_root, training
        )
        print(f"[CACHE] {status}: {cache_path}")
        if status == "HIT":
            archive_times.extend(timestamps)
        else:
            cache_misses.append(path)

    if missing_archives:
        print(
            f"[DATA] {len(missing_archives)} expected archive(s) are missing "
            "or empty; diagnostic does not download files."
        )

    cache_parent = cache_root
    while not os.path.exists(cache_parent):
        parent = os.path.dirname(cache_parent)
        if parent == cache_parent:
            break
        cache_parent = parent
    if os.path.exists(cache_parent):
        free_gib = shutil.disk_usage(cache_parent).free / (1024**3)
        print(f"[DISK] Free space at {cache_parent}: {free_gib:.2f} GiB")

    if args.max_open_files == 0:
        files_to_open = cache_misses
    else:
        files_to_open = cache_misses[:args.max_open_files]
    if len(cache_misses) > len(files_to_open):
        print(
            f"[GRIB] Inspecting {len(files_to_open)} of {len(cache_misses)} "
            "uncached archive(s); use --max-open-files 0 to inspect all."
        )

    open_failures = 0
    opened_files = set()
    for index, path in enumerate(files_to_open, 1):
        action = "open and regrid" if args.check_regrid else "open and inspect"
        print(
            f"[GRIB] {index}/{len(files_to_open)}: {action} {path} "
            f"(timeout {args.open_timeout:.0f}s; child uses indexpath='')",
            flush=True,
        )
        status, result, stdout, stderr, elapsed = _run_worker(
            path, bbox, args.res, args.open_timeout, args.check_regrid
        )
        if status == "TIMEOUT":
            open_failures += 1
            print(
                f"[GRIB] TIMEOUT after {elapsed:.1f}s: {path}. "
                "The isolated child was stopped; source GRIB/index files were "
                "not modified by cfgrib."
            )
            continue
        if result is None or status == "ERROR" or result.get("status") != "opened":
            open_failures += 1
            print(f"[GRIB] OPEN/REGRID ERROR after {elapsed:.1f}s: {path}")
            if result and result.get("error"):
                print("  " + result["error"])
            if stderr.strip():
                print("  Child stderr:")
                print("\n".join("    " + line for line in stderr.splitlines()[-12:]))
            continue

        opened_files.add(path)
        archive_times.extend(
            datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            for value in result["timestamps"]
        )
        print(
            f"[GRIB] OK in {elapsed:.1f}s: {result['groups']} dataset group(s), "
            f"{len(result['timestamps'])} valid time(s), "
            f"{len(result['variables'])} distinct field name(s)."
        )
        if result["missing_surface"]:
            print(
                "  [FIELDS] Missing required surface fields: "
                + ", ".join(result["missing_surface"])
            )
            open_failures += 1
        if result["missing_land"]:
            print(
                "  [FIELDS] Missing ERA5-Land fields (training will use defaults): "
                + ", ".join(result["missing_land"])
            )
        if result["missing_upper"]:
            print(
                "  [FIELDS] Missing upper-air fields/levels (defaults may be used): "
                + ", ".join(result["missing_upper"])
            )
        if result["source_bounds"]:
            print(f"  [GRID] Source coordinate bounds: {result['source_bounds']}")
        if args.check_regrid and result.get("regrid"):
            print(f"  [REGRID] {result['regrid']}")
        if stdout.strip() and "DEBUG_RESULT=" not in stdout:
            print("  [CHILD] " + " | ".join(stdout.splitlines()[-4:]))

    if archive_times:
        coverage_complete = (
            not missing_archives
            and len(files_to_open) == len(cache_misses)
            and len(opened_files) == len(files_to_open)
        )
        _report_time_coverage(archive_times, training, coverage_complete)
    elif cache_misses and not files_to_open:
        print(
            "[DATA] Timestamp coverage is partial because uncached files were "
            "not all inspected or could not be opened."
        )

    print("-" * 72)
    if open_failures or missing_archives:
        print(
            f"[SUMMARY] Issues found: {len(missing_archives)} missing/empty archive(s), "
            f"{open_failures} GRIB/required-field failure(s)."
        )
        return 1
    if len(files_to_open) < len(cache_misses):
        print(
            "[SUMMARY] Preflight is partial: selected archives/cache were checked, "
            "but some uncached GRIB files were not opened."
        )
        return 0
    print("[SUMMARY] Preflight checks completed without detected errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
