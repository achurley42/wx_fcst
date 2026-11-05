# get_operational_cds.py
import os
import argparse
from datetime import datetime, timedelta, timezone
import cdsapi

def download_operational_analysis(bbox, target_date_utc, output_file):
    """
    Downloads a single analysis timestamp (T0) from ERA5 via CDS API.
    Subsets geographically to the bounding box to keep file sizes minimal (<10 MB).
    """
    c = cdsapi.Client()
    
    # CDS Area format: [North, West, South, East]
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
    
    year = f"{target_date_utc.year:04d}"
    month = f"{target_date_utc.month:02d}"
    day = f"{target_date_utc.day:02d}"
    time_str = f"{target_date_utc.hour:02d}:00"
    
    temp_sfc = "temp_sfc.grib"
    temp_pl = "temp_pl.grib"

    print("=" * 70)
    print("  OPERATIONAL ERA5 DATA RETRIEVAL (CDS API)")
    print("=" * 70)
    print(f"[CDS] Target Timestamp : {year}-{month}-{day} {time_str} UTC")
    print(f"[CDS] Area Bounding Box: N={area[0]}, W={area[1]}, S={area[2]}, E={area[3]}")
    
    # 1. Download Surface and Static Fields
    print("[CDS] Requesting Single-Level Surface Variables & Static Orography/LSM...")
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
                "geopotential",  # Used for orography (elevation = z / 9.80665)
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": time_str,
            "area": area,
        },
        temp_sfc
    )

    # 2. Download Upper-Air Steering & Thermodynamic Levels
    print("[CDS] Requesting Upper-Air Pressure Levels (850, 700, 500, 250 hPa)...")
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

    # 3. Concatenate GRIB messages into a unified file
    print(f"[CDS] Merging surface and upper-air GRIB messages into: {output_file}")
    with open(output_file, "wb") as outfile:
        with open(temp_sfc, "rb") as sfc_f:
            outfile.write(sfc_f.read())
        with open(temp_pl, "rb") as pl_f:
            outfile.write(pl_f.read())

    # Cleanup temporary parts
    if os.path.exists(temp_sfc): os.remove(temp_sfc)
    if os.path.exists(temp_pl): os.remove(temp_pl)

    print(f"[SUCCESS] Operational GRIB2 file ready: {output_file} ({os.path.getsize(output_file)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download operational T0 ERA5 file from CDS API")
    parser.add_argument("--lat-min", type=float, required=True, help="South latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="North latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="West longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="East longitude")
    parser.add_argument("--output", type=str, default="./current_analysis.grib2", help="Output path")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD-HH format (defaults to 5 days ago to match ERA5 latency)")
    args = parser.parse_args()

    # ERA5 operational analysis typically has a 5-day release buffer; ERA5T has ~5 days
    if args.date:
        t_target = datetime.strptime(args.date, "%Y-%m-%d-%H").replace(tzinfo=timezone.utc)
    else:
        t_target = (datetime.now(timezone.utc) - timedelta(days=6)).replace(minute=0, second=0, microsecond=0)

    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max
    }
    download_operational_analysis(bbox, t_target, args.output)
