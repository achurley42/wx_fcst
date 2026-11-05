# get_historical_archive_cds.py
import os
import argparse
from datetime import datetime, timedelta, timezone
import cdsapi

def download_historical_archive(bbox, start_year, end_year, output_dir):
    """
    Downloads multi-month / multi-year 6-hourly ERA5 sequences for model training.
    """
    os.makedirs(output_dir, exist_ok=True)
    c = cdsapi.Client()
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
    
    print("=" * 70)
    print("  HISTORICAL TRAINING ARCHIVE RETRIEVAL (CDS API)")
    print("=" * 70)
    print(f"[CDS] Period: {start_year} through {end_year}")
    print(f"[CDS] Cadence: 6-hourly (00:00, 06:00, 12:00, 18:00 UTC)")
    print(f"[CDS] Target Directory: {output_dir}")

    years = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    times = ["00:00", "06:00", "12:00", "18:00"]

    for yr in years:
        for mo in months:
            out_sfc = os.path.join(output_dir, f"era5_sfc_{yr}_{mo}.grib")
            out_pl = os.path.join(output_dir, f"era5_pl_{yr}_{mo}.grib")
            merged_file = os.path.join(output_dir, f"era5_training_{yr}_{mo}.grib2")

            if os.path.exists(merged_file):
                print(f"[SKIP] Archive for {yr}-{mo} already exists on disk.")
                continue

            print(f"\n[CDS] Downloading Surface Archive for {yr}-{mo}...")
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
                out_sfc
            )

            print(f"[CDS] Downloading Pressure Level Archive for {yr}-{mo}...")
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
                out_pl
            )

            # Combine GRIB archives
            print(f"[CDS] Packaging combined training cube: {merged_file}")
            with open(merged_file, "wb") as outfile:
                with open(out_sfc, "rb") as f1: outfile.write(f1.read())
                with open(out_pl, "rb") as f2: outfile.write(f2.read())

            if os.path.exists(out_sfc): os.remove(out_sfc)
            if os.path.exists(out_pl): os.remove(out_pl)

    print("\n[SUCCESS] Historical archive download cycle complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical training GRIB2 data from CDS")
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--outdir", type=str, default="./historical_grib2")
    args = parser.parse_args()

    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max
    }
    download_historical_archive(bbox, args.start_year, args.end_year, args.outdir)
