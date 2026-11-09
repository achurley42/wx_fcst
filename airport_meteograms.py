import argparse
import csv
import math
import os
import re
from bisect import bisect_right
from datetime import datetime, timezone


_REQUIRED_FIELDS = ("t2m", "d2m", "u10", "v10", "msl", "tp_1h")
_OUTPUT_FIELDS = (
    "ICAO",
    "Airport_Name",
    "Airport_Latitude",
    "Airport_Longitude",
    "Forecast_Hour",
    "Valid_Time_UTC",
    "Temperature_2m_C",
    "Dewpoint_2m_C",
    "Wind_Speed_kt",
    "Wind_Direction_deg",
    "MSLP_hPa",
    "Precipitation_1h_mm",
)


def _read_airports(csv_path):
    airports = []
    seen_icaos = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as airport_file:
        reader = csv.DictReader(airport_file)
        if not reader.fieldnames:
            raise ValueError(f"Airport CSV has no header row: {csv_path}")
        columns = {
            column.strip().lower(): column
            for column in reader.fieldnames
            if column and column.strip()
        }
        missing_columns = {
            "icao",
            "name",
            "latitude",
            "longitude",
        } - set(columns)
        if missing_columns:
            raise ValueError(
                "Airport CSV must contain icao, name, latitude, longitude "
                f"columns; missing: {', '.join(sorted(missing_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            icao = (row.get(columns["icao"]) or "").strip().upper()
            name = (row.get(columns["name"]) or "").strip()
            try:
                latitude = float(row.get(columns["latitude"], ""))
                longitude = float(row.get(columns["longitude"], ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid airport coordinates on CSV line {line_number}."
                ) from exc
            if not re.fullmatch(r"[A-Z0-9]{4}", icao):
                raise ValueError(
                    f"Invalid four-character ICAO code on CSV line {line_number}: "
                    f"{icao!r}"
                )
            if icao in seen_icaos:
                raise ValueError(f"Duplicate ICAO code in airport CSV: {icao}")
            if (
                not math.isfinite(latitude)
                or not math.isfinite(longitude)
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
            ):
                raise ValueError(
                    f"Out-of-range airport coordinates on CSV line {line_number}."
                )
            seen_icaos.add(icao)
            airports.append(
                {
                    "icao": icao,
                    "name": name or icao,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

    if not airports:
        raise ValueError(f"Airport CSV contains no airport rows: {csv_path}")
    return airports


def validate_airport_csv(csv_path):
    return len(_read_airports(csv_path))


def _bracket(axis, value):
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = len(axis) - 1
        return last, last, 0.0
    upper = bisect_right(axis, value)
    lower = upper - 1
    fraction = (value - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def _grid_longitude(longitude, grid_longitudes):
    lower, upper = grid_longitudes[0], grid_longitudes[-1]
    candidates = [longitude + 360 * offset for offset in range(-2, 3)]
    within_grid = [candidate for candidate in candidates if lower <= candidate <= upper]
    if within_grid:
        return min(within_grid, key=lambda candidate: abs(candidate - longitude))
    return min(candidates, key=lambda candidate: abs(candidate - (lower + upper) / 2))


def _bilinear(values, time_index, lat_order, lon_order, lat_bracket, lon_bracket):
    lat0, lat1, fy = lat_bracket
    lon0, lon1, fx = lon_bracket
    y0, y1 = lat_order[lat0], lat_order[lat1]
    x0, x1 = lon_order[lon0], lon_order[lon1]
    lower = values[time_index][y0][x0] * (1 - fx) + values[time_index][y0][x1] * fx
    upper = values[time_index][y1][x0] * (1 - fx) + values[time_index][y1][x1] * fx
    return float(lower * (1 - fy) + upper * fy)


def _format_valid_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def export_airport_meteograms(netcdf_path, airport_csv, output_dir=None):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Airport meteograms require xarray in the selected Python environment."
        ) from exc

    netcdf_path = os.path.abspath(netcdf_path)
    airport_csv = os.path.abspath(airport_csv)
    if not os.path.isfile(netcdf_path):
        raise FileNotFoundError(f"Forecast NetCDF does not exist: {netcdf_path}")
    if not os.path.isfile(airport_csv):
        raise FileNotFoundError(f"Airport CSV does not exist: {airport_csv}")

    airports = _read_airports(airport_csv)
    output_dir = os.path.abspath(
        output_dir or os.path.join(os.path.dirname(netcdf_path), "airport_meteograms")
    )
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []
    with xr.open_dataset(netcdf_path) as dataset:
        for coordinate in ("time", "lat", "lon"):
            if coordinate not in dataset.coords:
                raise ValueError(
                    f"Forecast NetCDF is missing required coordinate: {coordinate}"
                )
        missing_fields = [field for field in _REQUIRED_FIELDS if field not in dataset]
        if missing_fields:
            raise ValueError(
                "Forecast NetCDF is missing required fields: "
                + ", ".join(missing_fields)
            )

        time_values = dataset["time"].values
        latitude_values = [float(value) for value in dataset["lat"].values]
        longitude_values = [float(value) for value in dataset["lon"].values]
        if not time_values.size or not latitude_values or not longitude_values:
            raise ValueError("Forecast NetCDF has an empty time or spatial coordinate.")
        if len(set(latitude_values)) != len(latitude_values) or len(
            set(longitude_values)
        ) != len(longitude_values):
            raise ValueError("Forecast NetCDF coordinates contain duplicate grid values.")

        lat_order = sorted(range(len(latitude_values)), key=latitude_values.__getitem__)
        lon_order = sorted(
            range(len(longitude_values)), key=longitude_values.__getitem__
        )
        sorted_lats = [latitude_values[index] for index in lat_order]
        sorted_lons = [longitude_values[index] for index in lon_order]
        arrays = {
            field: dataset[field].values
            for field in _REQUIRED_FIELDS
        }
        initialization = _format_valid_time(time_values[0])[:13].replace(" ", "T")
        initialization_tag = initialization.replace(":", "")

        for airport in airports:
            grid_lon = _grid_longitude(airport["longitude"], sorted_lons)
            if not (
                sorted_lats[0] <= airport["latitude"] <= sorted_lats[-1]
                and sorted_lons[0] <= grid_lon <= sorted_lons[-1]
            ):
                print(
                    f"[AIRPORT] WARNING: {airport['icao']} ({airport['name']}) at "
                    f"{airport['latitude']:.4f}, {airport['longitude']:.4f} is "
                    "outside the forecast grid; skipping."
                )
                continue

            lat_bracket = _bracket(sorted_lats, airport["latitude"])
            lon_bracket = _bracket(sorted_lons, grid_lon)
            safe_icao = re.sub(r"[^A-Z0-9_-]", "_", airport["icao"])
            output_path = os.path.join(
                output_dir,
                f"{safe_icao}_{initialization_tag}_meteogram.csv",
            )
            with open(output_path, "w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=_OUTPUT_FIELDS)
                writer.writeheader()
                for time_index, valid_time in enumerate(time_values):
                    values = {
                        field: _bilinear(
                            arrays[field],
                            time_index,
                            lat_order,
                            lon_order,
                            lat_bracket,
                            lon_bracket,
                        )
                        for field in _REQUIRED_FIELDS
                    }
                    wind_speed_kt = math.hypot(values["u10"], values["v10"]) * 1.94384
                    wind_direction = (
                        270 - math.degrees(math.atan2(values["v10"], values["u10"]))
                    ) % 360
                    writer.writerow(
                        {
                            "ICAO": airport["icao"],
                            "Airport_Name": airport["name"],
                            "Airport_Latitude": f"{airport['latitude']:.5f}",
                            "Airport_Longitude": f"{airport['longitude']:.5f}",
                            "Forecast_Hour": f"T+{time_index:02d}h",
                            "Valid_Time_UTC": _format_valid_time(valid_time),
                            "Temperature_2m_C": f"{values['t2m']:.2f}",
                            "Dewpoint_2m_C": f"{values['d2m']:.2f}",
                            "Wind_Speed_kt": f"{wind_speed_kt:.2f}",
                            "Wind_Direction_deg": int(round(wind_direction)) % 360,
                            "MSLP_hPa": f"{values['msl']:.2f}",
                            "Precipitation_1h_mm": f"{values['tp_1h']:.3f}",
                        }
                    )
            generated_files.append(output_path)
            print(f"[AIRPORT] Wrote {len(time_values)} forecast hour(s): {output_path}")

    if not generated_files:
        print("[AIRPORT] No airport meteograms were generated.")
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate airport meteogram CSVs from an existing forecast NetCDF."
    )
    parser.add_argument(
        "--netcdf",
        default="./forecast_output/forecast_72hr_surface_master.nc",
        help="Path to forecast_72hr_surface_master.nc.",
    )
    parser.add_argument(
        "--airports",
        required=True,
        help="CSV file with icao,name,latitude,longitude columns.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: <NetCDF directory>/airport_meteograms).",
    )
    args = parser.parse_args()
    export_airport_meteograms(args.netcdf, args.airports, args.output_dir)


if __name__ == "__main__":
    main()
