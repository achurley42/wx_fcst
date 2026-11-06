import json
import math
import os
import tempfile


SEASONS = ("DJF", "MAM", "JJA", "SON")
METEOROLOGICAL_SEASONS = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def season_for_time(valid_time, latitude):
    season = METEOROLOGICAL_SEASONS[valid_time.month]
    if latitude < 0:
        season = {"DJF": "JJA", "JJA": "DJF", "MAM": "SON", "SON": "MAM"}[season]
    return season


def _area(bbox):
    lat_min = math.radians(max(-90.0, bbox["lat_min"]))
    lat_max = math.radians(min(90.0, bbox["lat_max"]))
    lon_width = math.radians(bbox["lon_max"] - bbox["lon_min"])
    return max(0.0, math.sin(lat_max) - math.sin(lat_min)) * max(0.0, lon_width)


def overlap_fraction(requested_bbox, saved_bbox):
    intersection = {
        "lat_min": max(requested_bbox["lat_min"], saved_bbox["lat_min"]),
        "lat_max": min(requested_bbox["lat_max"], saved_bbox["lat_max"]),
        "lon_min": max(requested_bbox["lon_min"], saved_bbox["lon_min"]),
        "lon_max": min(requested_bbox["lon_max"], saved_bbox["lon_max"]),
    }
    requested_area = _area(requested_bbox)
    if requested_area == 0:
        return 0.0
    return min(1.0, _area(intersection) / requested_area)


def find_compatible_calibration(registry_path, bbox, season, minimum_overlap=0.5):
    if not os.path.isfile(registry_path):
        return None

    with open(registry_path, "r", encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    candidates = []
    for entry in registry.get("calibrations", []):
        if entry.get("season") != season or entry.get("input_channels") != 34:
            continue
        weight_path = entry.get("weights_path")
        if not weight_path or not os.path.isfile(weight_path):
            continue
        fraction = overlap_fraction(bbox, entry["bbox"])
        if fraction >= minimum_overlap:
            candidates.append((fraction, entry))
    if not candidates:
        return None
    fraction, entry = max(candidates, key=lambda candidate: candidate[0])
    return entry, fraction


def save_calibration(registry_path, calibration):
    registry_dir = os.path.dirname(os.path.abspath(registry_path))
    os.makedirs(registry_dir, exist_ok=True)

    if os.path.isfile(registry_path):
        with open(registry_path, "r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    else:
        registry = {"version": 1, "calibrations": []}

    entries = registry.setdefault("calibrations", [])
    entries[:] = [
        entry for entry in entries
        if not (
            entry.get("season") == calibration["season"]
            and entry.get("bbox") == calibration["bbox"]
        )
    ]
    entries.append(calibration)

    fd, temp_path = tempfile.mkstemp(prefix=".calibration-", suffix=".json", dir=registry_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=2, sort_keys=True)
            registry_file.write("\n")
        os.replace(temp_path, registry_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
