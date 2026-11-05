# run_forecast.py
import sys
import argparse
from datetime import datetime, timezone

# 1. Pre-Flight Dependency Verification (Runs before importing heavy packages)
try:
    from dependency_check import check_and_install_dependencies
    # Verify environment; prints status and auto-installs missing packages
    check_and_install_dependencies()
except Exception as e:
    print(f"[WARN] Dependency check encountered an issue: {e}. Attempting standard run...")

# 2. Main Application Imports
from ingest_grib2 import load_and_regrid_grib2
from model_architecture import HybridComputeManager, WeatherSpatioTemporalUNet
from forecast_engine import run_hourly_72hr_forecast, export_hourly_deliverables


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Hybrid GPU/CPU 72-Hour Hourly AI Weather Model")
    parser.add_argument("--grib", type=str, default="input_blended.grib2", help="Path to input GRIB2 file")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum Latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum Latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum Longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum Longitude")
    parser.add_argument("--res", type=float, default=0.10, help="Target spatial resolution in degrees")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to fine-tuned .pt weights")
    parser.add_argument("--plot-interval", type=int, default=6, help="Interval in hours for prognostic charts (e.g. 1, 6, 12)")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU")
    parser.add_argument("--skip-dep-check", action="store_true", help="Bypass pre-flight dependency checker")
    parser.add_argument("--outdir", type=str, default="./forecast_output", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }

    print("=" * 70)
    print("  AIR FORCE STANDALONE REGIONAL AI WEATHER MODEL (HOURLY ROLLOUT)")
    print("=" * 70)
    print(f"[CONFIG] Bounding Box : Lat [{bbox['lat_min']} to {bbox['lat_max']}], Lon [{bbox['lon_min']} to {bbox['lon_max']}]")
    print(f"[CONFIG] Resolution   : {args.res}° (~{args.res * 111.0:.1f} km)")
    print(f"[CONFIG] Cadence      : 1 Hour (73 total states: T+00h to T+72h)")
    print(f"[CONFIG] Output Dir   : {args.outdir}")

    # 1. Initialize Hybrid Hardware (CUDA GPU / MPS / CPU fallback)
    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)

    # 2. Ingest GRIB2 & Regrid to Target Domain
    grid_data, lats, lons = load_and_regrid_grib2(args.grib, bbox, args.res)

    # 3. Initialize Model & Execute Hourly Forecast
    model = WeatherSpatioTemporalUNet(in_channels=30, out_channels=6)
    init_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    history, valid_times = run_hourly_72hr_forecast(
        grid_data=grid_data,
        lats=lats,
        lons=lons,
        model=model,
        compute_mgr=compute_mgr,
        start_time_utc=init_time,
        weights_path=args.weights
    )

    # 4. Export Individual Hourly Files, Master NetCDF, CSV, and PNGs
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
    print("=" * 70)
    print(f"[COMPLETE] All hourly forecast deliverables written to: {args.outdir}")


if __name__ == "__main__":
    main()
