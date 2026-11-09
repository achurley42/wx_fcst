import glob
import os


def _format_bound(value):
    return format(float(value), ".10g")


def _archive_prefix(bbox):
    return (
        f"era5_analysis_lat{_format_bound(bbox['lat_min'])}_"
        f"{_format_bound(bbox['lat_max'])}_"
        f"lon{_format_bound(bbox['lon_min'])}_"
        f"{_format_bound(bbox['lon_max'])}"
    )


def operational_archive_path(input_dir, bbox, valid_time_utc):
    filename = f"{_archive_prefix(bbox)}_{valid_time_utc:%Y_%m_%d_%H}.grib2"
    return os.path.join(input_dir, filename)


def latest_operational_archive(input_dir, bbox):
    matches = glob.glob(
        os.path.join(input_dir, f"{_archive_prefix(bbox)}_*.grib2")
    )
    return max(matches) if matches else None
