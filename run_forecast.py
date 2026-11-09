# run_forecast.py
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

# 1. Pre-Flight Dependency Verification
try:
    from dependency_check import check_and_install_dependencies
    check_and_install_dependencies()
except Exception as e:
    print(f"[WARN] Dependency check encountered an issue: {e}. Attempting standard run...")

from ingest_grib2 import ERA5_LAND_ALIASES, load_and_regrid_grib2
from calibration_registry import find_compatible_calibration
from model_architecture import (
    MODEL_INPUT_CHANNELS,
    HybridComputeManager,
    WeatherSpatioTemporalUNet,
)
from forecast_engine import run_hourly_72hr_forecast, export_hourly_deliverables
from forecast_dashboard import write_forecast_dashboard
from operational_archive import latest_operational_archive, operational_archive_path
from airport_meteograms import export_airport_meteograms, validate_airport_csv


def retrieve_cds_operational_analysis(bbox, target_date_utc, output_path):
    """
    Downloads a single T0 analysis timestamp via CDS API restricted to the user's bounding box.
    """
    try:
        import cdsapi
    except ImportError:
        print("[ERROR] 'cdsapi' library is missing. Install with: pip install cdsapi")
        sys.exit(1)

    c = cdsapi.Client()
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]

    year = f"{target_date_utc.year:04d}"
    month = f"{target_date_utc.month:02d}"
    day = f"{target_date_utc.day:02d}"
    time_str = f"{target_date_utc.hour:02d}:00"

    temp_sfc = output_path + ".sfc.tmp"
    temp_pl = output_path + ".pl.tmp"
    temp_land = output_path + ".land.tmp"

    print("=" * 70)
    print("  AUTOMATED OPERATIONAL CDS API RETRIEVAL (T0 ANALYSIS)")
    print("=" * 70)
    print(f"[CDS] Target Timestamp : {year}-{month}-{day} {time_str} UTC")
    print(f"[CDS] Area Bounding Box: N={area[0]}, W={area[1]}, S={area[2]}, E={area[3]}")

    print("[CDS] Pulling Surface Variables & Topography/LSM...")
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
            "year": year,
            "month": month,
            "day": day,
            "time": time_str,
            "area": area,
        },
        temp_sfc
    )

    print("[CDS] Pulling Isobaric Levels (850, 700, 500, 250 hPa)...")
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
            "year": year,
            "month": month,
            "day": day,
            "time": time_str,
            "area": area,
        },
        temp_pl
    )

    print("[CDS] Pulling ERA5-Land skin, soil, moisture, and snow fields...")
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
            "year": year,
            "month": month,
            "day": day,
            "time": time_str,
            "area": area,
        },
        temp_land
    )

    print(f"[CDS] Packaging combined ERA5 and ERA5-Land GRIB2 analysis: {output_path}")
    with open(output_path, "wb") as outfile:
        for temp_path in (temp_sfc, temp_pl, temp_land):
            with open(temp_path, "rb") as data_file:
                outfile.write(data_file.read())

    for temp_path in (temp_sfc, temp_pl, temp_land):
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print(f"[SUCCESS] Download completed ({os.path.getsize(output_path)/1024/1024:.2f} MB).\n")


def existing_cds_data_matches_time(grib_path, target_time_utc):
    if not os.path.isfile(grib_path) or os.path.getsize(grib_path) == 0:
        return False

    datasets = []
    try:
        import cfgrib
        import numpy as np

        datasets = cfgrib.open_datasets(grib_path)
        if not datasets:
            print(f"[CDS] Existing file has no readable GRIB datasets: {grib_path}")
            return False

        expected_time = np.datetime64(
            target_time_utc.astimezone(timezone.utc).replace(tzinfo=None), "s"
        )
        for dataset in datasets:
            time_coord = dataset.coords.get("time", dataset.coords.get("valid_time"))
            if time_coord is None:
                print(f"[CDS] Existing GRIB data has no timestamp: {grib_path}")
                return False

            actual_times = np.asarray(time_coord.values).astype("datetime64[s]")
            if actual_times.size == 0 or np.any(actual_times != expected_time):
                print(
                    f"[CDS] Existing GRIB timestamp does not match "
                    f"{target_time_utc.strftime('%Y-%m-%d %H:%M UTC')}: {grib_path}"
                )
                return False
        available_vars = set().union(*(dataset.data_vars for dataset in datasets))
        if not all(available_vars.intersection(aliases) for aliases in ERA5_LAND_ALIASES):
            print(f"[CDS] Existing GRIB is missing ERA5-Land fields: {grib_path}")
            return False
        return True
    except Exception as exc:
        print(f"[CDS] Could not verify existing GRIB data ({exc}); downloading again.")
        return False
    finally:
        for dataset in datasets:
            dataset.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Hybrid GPU/CPU 72-Hour Hourly AI Weather Model")
    parser.add_argument(
        "--grib",
        type=str,
        default=None,
        help=(
            "Path to input GRIB2 file. With --use-cds, defaults to a "
            "bounds/date-named file under ./input_data; otherwise uses the "
            "latest matching archive there."
        ),
    )
    parser.add_argument("--use-cds", action="store_true", help="Automatically pull operational T0 analysis via CDS API")
    parser.add_argument("--cds-date", type=str, default=None, help="Target UTC date for CDS pull (YYYY-MM-DD-HH). Defaults to 6 days ago (ERA5 release buffer)")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum Latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum Latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum Longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum Longitude")
    parser.add_argument("--res", type=float, default=0.10, help="Target spatial resolution in degrees")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to fine-tuned .pt weights")
    parser.add_argument(
        "--calibration-registry",
        type=str,
        default="./checkpoints/calibration_registry.json",
        help="Regional/monthly calibrated-weight registry",
    )
    parser.add_argument("--plot-interval", type=int, default=1, help="Interval in hours for prognostic charts (e.g. 1, 6, 12)")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU")
    parser.add_argument("--skip-dep-check", action="store_true", help="Bypass pre-flight dependency checker")
    parser.add_argument("--outdir", type=str, default="./forecast_output", help="Output directory")
    parser.add_argument(
        "--airports",
        type=str,
        default=None,
        help=(
            "Optional airport CSV (icao,name,latitude,longitude) for "
            "bilinearly interpolated per-airport meteograms."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.airports:
        airport_count = validate_airport_csv(args.airports)
        print(f"[AIRPORT] Validated {airport_count} airport(s) from {args.airports}.")
    run_started_at = datetime.now(timezone.utc)
    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }

    # 1. Automated CDS API Ingestion Hook
    if args.use_cds:
        if args.cds_date:
            target_time = datetime.strptime(args.cds_date, "%Y-%m-%d-%H").replace(tzinfo=timezone.utc)
        else:
            # ERA5 full reanalysis typically has a 5-day release buffer
            target_time = (datetime.now(timezone.utc) - timedelta(days=6)).replace(minute=0, second=0, microsecond=0)

        if args.grib is None:
            args.grib = operational_archive_path("./input_data", bbox, target_time)
        os.makedirs(os.path.dirname(os.path.abspath(args.grib)), exist_ok=True)
        if existing_cds_data_matches_time(args.grib, target_time):
            print(
                f"[CDS] Reusing existing data for "
                f"{target_time.strftime('%Y-%m-%d %H:%M UTC')}: {args.grib}"
            )
        else:
            retrieve_cds_operational_analysis(bbox, target_time, args.grib)
    elif args.grib is None:
        args.grib = latest_operational_archive("./input_data", bbox)
        if args.grib is None:
            raise FileNotFoundError(
                "No bounds-qualified operational GRIB was found under "
                "./input_data. Specify --grib PATH or use --use-cds to "
                "download the analysis."
            )

    print("=" * 70)
    print("  AIR FORCE STANDALONE REGIONAL AI WEATHER MODEL (HOURLY ROLLOUT)")
    print("=" * 70)
    print(f"[CONFIG] Bounding Box : Lat [{bbox['lat_min']} to {bbox['lat_max']}], Lon [{bbox['lon_min']} to {bbox['lon_max']}]")
    print(f"[CONFIG] Resolution   : {args.res}° (~{args.res * 111.0:.1f} km)")
    print(f"[CONFIG] Cadence      : 1 Hour (73 total states: T+00h to T+72h)")
    print(f"[CONFIG] Output Dir   : {args.outdir}")

    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)
    grid_data, lats, lons = load_and_regrid_grib2(args.grib, bbox, args.res)

    model = WeatherSpatioTemporalUNet(in_channels=MODEL_INPUT_CHANNELS, out_channels=6)
    init_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    weights_path = args.weights
    weights_source = "explicit --weights argument" if args.weights else None
    calibration = None
    calibration_overlap = None
    if weights_path is None:
        match = find_compatible_calibration(
            args.calibration_registry, bbox, init_time.month, minimum_overlap=0.5
        )
        if match:
            calibration, calibration_overlap = match
            weights_path = calibration["weights_path"]
            weights_source = "monthly calibration registry"
            print(
                f"[MODEL] Reusing {init_time.strftime('%B')} calibration from "
                f"{calibration['bbox']} (region overlap {calibration_overlap:.0%}, "
                f"status={calibration.get('status', 'accepted')}): {weights_path}"
            )
        else:
            weights_source = "baseline (no matching calibration)"
            print(
                f"[MODEL] No {init_time.strftime('%B')} calibration overlaps this "
                "region by at least 50%; "
                "using baseline weights."
            )

    history, valid_times = run_hourly_72hr_forecast(
        grid_data=grid_data,
        lats=lats,
        lons=lons,
        model=model,
        compute_mgr=compute_mgr,
        start_time_utc=init_time,
        weights_path=weights_path
    )

    export_hourly_deliverables(
        history=history,
        lats=lats,
        lons=lons,
        valid_times=valid_times,
        orog=grid_data["orog"],
        lsm=grid_data["lsm"],
        output_dir=args.outdir,
        plot_interval=args.plot_interval
    )
    airport_meteogram_files = []
    if args.airports:
        airport_meteogram_files = export_airport_meteograms(
            os.path.join(args.outdir, "forecast_72hr_surface_master.nc"),
            args.airports,
        )
    effective_weights_path = (
        os.path.abspath(weights_path)
        if weights_path and os.path.isfile(weights_path)
        else None
    )
    if args.weights and effective_weights_path is None:
        weights_source = "baseline (requested checkpoint was not found)"
    input_grib_path = os.path.abspath(args.grib)
    input_grib_size = (
        os.path.getsize(input_grib_path) if os.path.isfile(input_grib_path) else None
    )
    if input_grib_size is None:
        input_grib_status = "Missing; synthetic grid fallback was used"
    else:
        input_grib_status = (
            "Present; an unreadable GRIB can trigger the synthetic-grid fallback"
        )
    metadata = {
        "run_started_utc": run_started_at.isoformat(timespec="seconds"),
        "forecast_initialization_utc": init_time.isoformat(timespec="seconds"),
        "forecast_valid_start_utc": valid_times[0].isoformat(timespec="minutes"),
        "forecast_valid_end_utc": valid_times[-1].isoformat(timespec="minutes"),
        "forecast_duration_hours": 72,
        "timestep_hours": 1,
        "forecast_states": len(valid_times),
        "source_data_mode": "CDS download/reuse" if args.use_cds else "local GRIB",
        "input_grib_file": input_grib_path,
        "input_grib_status": input_grib_status,
        "input_grib_size_bytes": input_grib_size,
        "requested_area": (
            f"Lat {bbox['lat_min']} to {bbox['lat_max']}, "
            f"lon {bbox['lon_min']} to {bbox['lon_max']}"
        ),
        "grid_area": (
            f"Lat {float(lats[0]):.6g} to {float(lats[-1]):.6g}, "
            f"lon {float(lons[0]):.6g} to {float(lons[-1]):.6g}"
        ),
        "grid_shape_lat_lon": f"{len(lats)} x {len(lons)}",
        "requested_resolution_degrees": args.res,
        "approximate_resolution_km": round(args.res * 111.0, 2),
        "model_architecture": "WeatherSpatioTemporalUNet",
        "model_mode": (
            "Fine-tuned model" if effective_weights_path else "Hybrid baseline (no trained checkpoint)"
        ),
        "model_input_channels": MODEL_INPUT_CHANNELS,
        "model_output_channels": 6,
        "weights_source": weights_source,
        "weights_requested_path": os.path.abspath(args.weights) if args.weights else "not specified",
        "weights_loaded_from": effective_weights_path or "None; baseline weights used",
        "calibration_status": (
            calibration.get("status", "accepted") if calibration else "not applicable"
        ),
        "calibration_month": calibration.get("month") if calibration else None,
        "calibration_region": (
            str(calibration.get("bbox")) if calibration else "not applicable"
        ),
        "calibration_validation_score": (
            calibration.get("validation_score") if calibration else None
        ),
        "calibration_region_overlap": (
            f"{calibration_overlap:.0%}" if calibration_overlap is not None else "not applicable"
        ),
        "compute_device": str(compute_mgr.device),
        "forced_cpu": args.force_cpu,
        "chart_interval_hours": args.plot_interval,
        "airport_list_csv": (
            os.path.abspath(args.airports) if args.airports else "not specified"
        ),
        "airport_meteogram_count": len(airport_meteogram_files),
        "output_directory": os.path.abspath(args.outdir),
    }
    dashboard_path, metadata_path = write_forecast_dashboard(
        output_dir=args.outdir,
        metadata=metadata,
        inventory_root=os.path.dirname(os.path.abspath(__file__)),
        extra_files=(
            input_grib_path,
            effective_weights_path,
            args.calibration_registry,
            args.airports,
        ),
    )
    print(f"[EXPORT] Forecast dashboard: {dashboard_path}")
    print(f"[EXPORT] Forecast metadata: {metadata_path}")
    print("=" * 70)
    print(f"[COMPLETE] All hourly forecast deliverables written to: {args.outdir}")


if __name__ == "__main__":
    main()
