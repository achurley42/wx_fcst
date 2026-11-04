# run_forecast.py
import argparse
from datetime import datetime, timezone
from ingest_grib2 import load_and_regrid_grib2
from model_architecture import HybridComputeManager, WeatherSpatioTemporalUNet
from forecast_engine import run_72hr_forecast, export_deliverables


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Hybrid GPU/CPU 72-Hour Regional AI Weather Model")
    parser.add_argument("--grib", type=str, default="input_blended.grib2", help="Path to input ERA5/Blended GRIB2 file")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum Latitude (e.g., 25.0)")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum Latitude (e.g., 35.0)")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum Longitude (e.g., -90.0)")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum Longitude (e.g., -75.0)")
    parser.add_argument("--res", type=float, default=0.10, help="Target spatial resolution in degrees (e.g., 0.10 for ~11km, 0.05 for ~5.5km)")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to fine-tuned PyTorch .pt weights")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU even if GPU is present")
    parser.add_argument("--outdir", type=str, default="./forecast_output", help="Output directory for NetCDF, CSV, and PNGs")
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
    print("  STANDALONE REGIONAL AI WEATHER FORECAST SYSTEM (72-HOUR ROLLOUT)")
    print("=" * 70)
    print(f"[CONFIG] Bounding Box: Lat [{bbox['lat_min']} to {bbox['lat_max']}], Lon [{bbox['lon_min']} to {bbox['lon_max']}]")
    print(f"[CONFIG] Target Spatial Resolution: {args.res}° (~{args.res * 111.0:.1f} km)")

    # 1. Initialize Hybrid GPU/CPU Compute Manager
    compute_mgr = HybridComputeManager(force_cpu=args.force_cpu)

    # 2. Ingest GRIB2 & Regrid to User-Defined Domain
    grid_data, lats, lons = load_and_regrid_grib2(args.grib, bbox, args.res)
    print(f"[INGEST] Regridded Domain Shape: {len(lats)} Lat x {len(lons)} Lon grid points.")

    # 3. Initialize Model & Run 72-Hour Rollout
    model = WeatherSpatioTemporalUNet(in_channels=30, out_channels=6)
    init_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    history, valid_times = run_72hr_forecast(
        grid_data=grid_data,
        lats=lats,
        lons=lons,
        model=model,
        compute_mgr=compute_mgr,
        start_time_utc=init_time,
        weights_path=args.weights
    )

    # 4. Export NetCDF4, CSV, and 4-Panel Surface Charts
    export_deliverables(
        history=history,
        lats=lats,
        lons=lons,
        valid_times=valid_times,
        orog=grid_data["orog"],
        lsm=grid_data["lsm"],
        output_dir=args.outdir
    )
    print("=" * 70)
    print(f"[COMPLETE] All 72-hour deliverables written to: {args.outdir}")


if __name__ == "__main__":
    main()
