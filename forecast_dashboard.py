import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time
from urllib.parse import quote


_SKIP_DIRECTORIES = {".git", ".cache", ".venv", "__pycache__", "venv"}
_GRIB_EXTENSIONS = {".grb", ".grb2", ".grib", ".grib2"}


def _parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _infer_metadata(output_dir, forecast_init=None):
    master_path = os.path.join(output_dir, "forecast_72hr_surface_master.nc")
    initial_path = os.path.join(
        output_dir, "hourly_nc", "forecast_surface_T+00h.nc"
    )
    netcdf_path = master_path if os.path.isfile(master_path) else initial_path
    if not os.path.isfile(netcdf_path):
        raise FileNotFoundError(
            "Cannot infer forecast metadata: expected "
            f"{master_path} or {initial_path}. Specify --forecast-init only "
            "if you can provide the initialization time, but forecast NetCDF "
            "output is still needed for area/grid metadata."
        )

    try:
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Reading existing forecast NetCDF files requires xarray and numpy."
        ) from exc

    with xr.open_dataset(netcdf_path) as dataset:
        if "time" in dataset.coords and dataset["time"].size:
            times = np.asarray(dataset["time"].values).reshape(-1)
            time_values = [
                _parse_utc(np.datetime_as_string(value, unit="s"))
                for value in times
            ]
        elif "valid_time_utc" in dataset.attrs:
            time_values = [_parse_utc(dataset.attrs["valid_time_utc"])]
        else:
            time_values = []

        latitudes = np.asarray(dataset["lat"].values) if "lat" in dataset.coords else None
        longitudes = np.asarray(dataset["lon"].values) if "lon" in dataset.coords else None

    if len(time_values) <= 1:
        hourly_dir = os.path.join(output_dir, "hourly_nc")
        for filename in os.listdir(hourly_dir) if os.path.isdir(hourly_dir) else ():
            match = re.fullmatch(r"forecast_surface_T\+(\d+)h\.nc", filename)
            if not match:
                continue
            with xr.open_dataset(os.path.join(hourly_dir, filename)) as dataset:
                valid_time = dataset.attrs.get("valid_time_utc")
            if valid_time:
                time_values.append(_parse_utc(valid_time))
        time_values = sorted(set(time_values))

    initialization = _parse_utc(forecast_init) if forecast_init else (
        time_values[0] if time_values else None
    )
    if initialization is None:
        raise ValueError(
            "Forecast NetCDF has no usable time coordinate. Provide "
            "--forecast-init YYYY-MM-DDTHH:MM:SSZ."
        )
    valid_times = time_values or [initialization]
    valid_times = sorted(valid_times)

    metadata = {
        "metadata_source": "Inferred from existing forecast NetCDF; original run settings were not saved.",
        "run_started_utc": "not recorded",
        "forecast_initialization_utc": initialization.isoformat(timespec="seconds"),
        "forecast_valid_start_utc": valid_times[0].isoformat(timespec="minutes"),
        "forecast_valid_end_utc": valid_times[-1].isoformat(timespec="minutes"),
        "forecast_duration_hours": round(
            (valid_times[-1] - valid_times[0]).total_seconds() / 3600
        ),
        "timestep_hours": (
            round((valid_times[1] - valid_times[0]).total_seconds() / 3600, 6)
            if len(valid_times) > 1
            else "not recorded"
        ),
        "forecast_states": len(valid_times),
        "source_data_mode": "not recorded",
        "input_grib_file": "not recorded",
        "input_grib_status": "not recorded",
        "input_grib_size_bytes": None,
        "requested_area": "not recorded; grid bounds inferred below",
        "grid_area": (
            f"Lat {float(np.nanmin(latitudes)):.6g} to {float(np.nanmax(latitudes)):.6g}, "
            f"lon {float(np.nanmin(longitudes)):.6g} to {float(np.nanmax(longitudes)):.6g}"
            if latitudes is not None and longitudes is not None
            else "not recorded"
        ),
        "grid_shape_lat_lon": (
            f"{len(latitudes)} x {len(longitudes)}"
            if latitudes is not None and longitudes is not None
            else "not recorded"
        ),
        "requested_resolution_degrees": (
            round(float(np.median(np.abs(np.diff(latitudes)))), 8)
            if latitudes is not None and len(latitudes) > 1
            else "not recorded"
        ),
        "approximate_resolution_km": (
            round(float(np.median(np.abs(np.diff(latitudes)))) * 111.0, 2)
            if latitudes is not None and len(latitudes) > 1
            else "not recorded"
        ),
        "model_architecture": "not recorded",
        "model_mode": "not recorded for this prior run",
        "model_input_channels": "not recorded",
        "model_output_channels": "not recorded",
        "weights_source": "not recorded",
        "weights_requested_path": "not recorded",
        "weights_loaded_from": "not recorded",
        "calibration_status": "not recorded",
        "calibration_month": None,
        "calibration_region": "not recorded",
        "calibration_validation_score": None,
        "calibration_region_overlap": "not recorded",
        "compute_device": "not recorded",
        "forced_cpu": "not recorded",
        "chart_interval_hours": "inferred from available chart images",
        "output_directory": os.path.abspath(output_dir),
    }
    return metadata


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a forecast dashboard from existing forecast outputs "
            "without rerunning the forecast."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="./forecast_output",
        help="Existing forecast output directory (default: ./forecast_output).",
    )
    parser.add_argument(
        "--inventory-root",
        default=".",
        help="Directory tree to include in the file inventory (default: current directory).",
    )
    parser.add_argument(
        "--forecast-init",
        help="Initialization time override, ISO-8601 UTC (e.g. 2026-11-09T06:00:00Z).",
    )
    parser.add_argument(
        "--input-grib",
        help="Optional input GRIB file to include in the inventory.",
    )
    parser.add_argument(
        "--weights",
        help="Optional model checkpoint to include in the inventory.",
    )
    return parser.parse_args()


def _file_category(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in _GRIB_EXTENSIONS:
        return "GRIB"
    if extension in {".pt", ".pth", ".ckpt"}:
        return "Model weights"
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "Image"
    if extension in {".nc", ".nc4", ".netcdf"}:
        return "NetCDF"
    if extension in {".csv", ".json", ".log"}:
        return "Data / logs"
    if extension == ".html":
        return "Dashboard"
    if extension in {".py", ".sh", ".md", ".yml", ".yaml", ".toml"}:
        return "Code / documentation"
    return "Other"


def _inventory_files(inventory_roots, extra_paths, dashboard_dir):
    if isinstance(inventory_roots, (str, os.PathLike)):
        inventory_roots = (inventory_roots,)
    paths = set()
    def report_walk_error(error):
        print(f"[DASHBOARD] Could not scan inventory directory: {error}")

    for inventory_root in inventory_roots:
        for current_root, directories, filenames in os.walk(
            os.path.abspath(inventory_root), onerror=report_walk_error
        ):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory.lower() not in _SKIP_DIRECTORIES
            )
            for filename in filenames:
                paths.add(os.path.abspath(os.path.join(current_root, filename)))

    for path in extra_paths:
        if path and os.path.isfile(path):
            paths.add(os.path.abspath(path))

    entries = []
    for path in sorted(paths, key=str.casefold):
        try:
            stat = os.stat(path)
            relative_path = os.path.relpath(path, dashboard_dir).replace(os.sep, "/")
            href = quote(relative_path, safe="/:+-._~")
        except ValueError:
            relative_path = path
            href = Path(path).as_uri()
        except OSError as exc:
            print(f"[DASHBOARD] Could not inspect inventory file {path}: {exc}")
            continue
        entries.append(
            {
                "name": os.path.basename(path),
                "path": relative_path,
                "href": href,
                "category": _file_category(path),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return entries


def _dashboard_html(data):
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Weather Forecast Run Dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, Segoe UI, sans-serif; background: #101923; color: #e7eef5; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header { padding: 1.4rem max(1rem, calc((100vw - 1440px) / 2)); background: #172635; border-bottom: 1px solid #31495d; }
    h1 { margin: 0; font-size: 1.5rem; }
    header p { margin: .4rem 0 0; color: #a9bdce; }
    main { max-width: 1440px; margin: auto; padding: 1rem; display: grid; gap: 1rem; }
    section { background: #172635; border: 1px solid #31495d; border-radius: 10px; padding: 1rem; min-width: 0; }
    h2 { font-size: 1.1rem; margin: 0 0 .8rem; }
    .metadata { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .7rem; }
    .card { padding: .8rem; background: #203447; border-radius: 7px; overflow-wrap: anywhere; }
    .label { display: block; color: #a9bdce; font-size: .78rem; margin-bottom: .25rem; }
    .value { font-weight: 600; }
    .slider-head, .controls { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
    .slider-head { justify-content: space-between; margin-bottom: .6rem; }
    #chart { display: block; max-width: 100%; max-height: 75vh; width: auto; height: auto; margin: 0 auto; background: #0b1118; }
    input[type="range"] { flex: 1; min-width: 180px; accent-color: #50b6d8; }
    button, select, input[type="search"] { background: #263d51; color: #e7eef5; border: 1px solid #49647b; border-radius: 5px; padding: .5rem .7rem; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: default; }
    .table-wrap { overflow: auto; max-height: 55vh; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th, td { padding: .55rem; text-align: left; border-bottom: 1px solid #31495d; }
    th { position: sticky; top: 0; background: #203447; }
    td a { color: #74c8e5; }
    .inventory-tools { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: .6rem; }
    #inventory-count, .muted { color: #a9bdce; font-size: .85rem; }
    .empty { color: #f0c674; padding: 1rem; text-align: center; }
    @media (min-width: 1050px) { main { grid-template-columns: minmax(0, 1.7fr) minmax(340px, 1fr); } .metadata-section, .inventory-section { grid-column: 1 / -1; } }
  </style>
</head>
<body>
  <header>
    <h1>Weather Forecast Run Dashboard</h1>
    <p id="run-subtitle"></p>
  </header>
  <main>
    <section class="metadata-section">
      <h2>Run metadata</h2>
      <div id="metadata" class="metadata"></div>
    </section>
    <section>
      <div class="slider-head">
        <h2>Forecast charts</h2>
        <span id="chart-caption" class="muted"></span>
      </div>
      <div class="controls">
        <button id="previous" type="button" aria-label="Previous chart">Previous</button>
        <input id="chart-slider" type="range" min="0" value="0" step="1" aria-label="Forecast hour">
        <button id="next" type="button" aria-label="Next chart">Next</button>
      </div>
      <div id="chart-empty" class="empty" hidden>No forecast chart images were found.</div>
      <img id="chart" alt="Forecast chart">
    </section>
    <section class="inventory-section">
      <h2>Available files</h2>
      <div class="inventory-tools">
        <input id="file-search" type="search" placeholder="Filter by file name or path" aria-label="Filter inventory">
        <select id="category-filter" aria-label="Filter by file type"><option value="">All file types</option></select>
        <span id="inventory-count"></span>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>File</th><th>Type</th><th>Size</th><th>Modified (local)</th></tr></thead>
          <tbody id="inventory"></tbody>
        </table>
      </div>
      <p class="muted">Inventory includes files found under the project directory and the run's input and model-weight files.</p>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    "use strict";
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const metadata = document.getElementById("metadata");
    const display = (value) => value === null || value === undefined || value === "" ? "Not available" : String(value);
    document.getElementById("run-subtitle").textContent =
      `${display(data.metadata.forecast_initialization_utc)} · ${display(data.metadata.model_mode)}`;

    for (const [label, value] of Object.entries(data.metadata)) {
      const card = document.createElement("div");
      card.className = "card";
      const heading = document.createElement("span");
      heading.className = "label";
      heading.textContent = label.replace(/_/g, " ");
      const content = document.createElement("span");
      content.className = "value";
      content.textContent = display(value);
      card.append(heading, content);
      metadata.append(card);
    }

    const charts = data.charts;
    const slider = document.getElementById("chart-slider");
    const image = document.getElementById("chart");
    const caption = document.getElementById("chart-caption");
    const empty = document.getElementById("chart-empty");
    slider.max = String(Math.max(0, charts.length - 1));
    slider.disabled = charts.length < 2;
    document.getElementById("previous").disabled = charts.length === 0;
    document.getElementById("next").disabled = charts.length === 0;
    empty.hidden = charts.length > 0;
    image.hidden = charts.length === 0;
    function showChart(index) {
      if (!charts.length) return;
      const chart = charts[index];
      slider.value = String(index);
      image.src = chart.href;
      image.alt = `Forecast chart ${chart.forecast_hour}`;
      caption.textContent = `${chart.forecast_hour} · valid ${chart.valid_time_utc}`;
    }
    slider.addEventListener("input", () => showChart(Number(slider.value)));
    document.getElementById("previous").addEventListener("click", () => {
      showChart((Number(slider.value) + charts.length - 1) % charts.length);
    });
    document.getElementById("next").addEventListener("click", () => {
      showChart((Number(slider.value) + 1) % charts.length);
    });
    showChart(0);

    const search = document.getElementById("file-search");
    const category = document.getElementById("category-filter");
    const inventoryBody = document.getElementById("inventory");
    const count = document.getElementById("inventory-count");
    for (const type of [...new Set(data.files.map((file) => file.category))].sort()) {
      const option = document.createElement("option");
      option.value = type;
      option.textContent = type;
      category.append(option);
    }
    function renderInventory() {
      const query = search.value.toLocaleLowerCase();
      const selectedCategory = category.value;
      const filtered = data.files.filter((file) =>
        (!selectedCategory || file.category === selectedCategory) &&
        `${file.name} ${file.path}`.toLocaleLowerCase().includes(query)
      );
      inventoryBody.replaceChildren();
      for (const file of filtered) {
        const row = document.createElement("tr");
        const name = document.createElement("td");
        const link = document.createElement("a");
        link.href = file.href;
        link.textContent = file.path;
        name.append(link);
        const type = document.createElement("td");
        type.textContent = file.category;
        const size = document.createElement("td");
        size.textContent = formatSize(file.size);
        const modified = document.createElement("td");
        modified.textContent = new Date(file.modified * 1000).toLocaleString();
        row.append(name, type, size, modified);
        inventoryBody.append(row);
      }
      count.textContent = `${filtered.length} of ${data.files.length} files`;
    }
    function formatSize(size) {
      if (size < 1024) return `${size} B`;
      const units = ["KiB", "MiB", "GiB", "TiB"];
      let amount = size / 1024;
      let index = 0;
      while (amount >= 1024 && index < units.length - 1) {
        amount /= 1024;
        index += 1;
      }
      return `${amount.toFixed(1)} ${units[index]}`;
    }
    search.addEventListener("input", renderInventory);
    category.addEventListener("change", renderInventory);
    renderInventory();
  </script>
</body>
</html>
""".replace("__DASHBOARD_DATA__", payload)


def write_forecast_dashboard(output_dir, metadata, inventory_root, extra_files=()):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dashboard_path = os.path.join(output_dir, "forecast_dashboard.html")
    metadata_path = os.path.join(output_dir, "forecast_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    charts = []
    for filename in os.listdir(output_dir):
        match = re.fullmatch(r"surface_prog_T\+(\d+)h\.png", filename)
        if not match:
            continue
        hour = int(match.group(1))
        initialization = datetime.fromisoformat(
            metadata["forecast_initialization_utc"].replace("Z", "+00:00")
        )
        valid_time = initialization + timedelta(hours=hour)
        charts.append(
            {
                "forecast_hour": f"T+{hour:02d}h",
                "valid_time_utc": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
                "href": quote(filename, safe="/:+-._~"),
            }
        )
    charts.sort(key=lambda chart: chart["forecast_hour"])
    dashboard_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "forecast_initialization_utc_dt"
    }
    dashboard_metadata["dashboard_file"] = os.path.basename(dashboard_path)
    dashboard_metadata["metadata_file"] = os.path.basename(metadata_path)

    files = _inventory_files(
        (inventory_root, output_dir),
        [*extra_files, metadata_path],
        output_dir,
    )
    files = [
        entry for entry in files
        if entry["path"] != os.path.basename(dashboard_path)
    ]
    files.append(
        {
            "name": os.path.basename(dashboard_path),
            "path": os.path.basename(dashboard_path),
            "href": quote(os.path.basename(dashboard_path), safe="/:+-._~"),
            "category": "Dashboard",
            "size": 0,
            "modified": time(),
        }
    )
    data = {"metadata": dashboard_metadata, "charts": charts, "files": files}
    page = _dashboard_html(data)
    for _ in range(3):
        size = len(page.encode("utf-8"))
        if files[-1]["size"] == size:
            break
        files[-1]["size"] = size
        page = _dashboard_html(data)
    with open(dashboard_path, "w", encoding="utf-8", newline="") as dashboard_file:
        dashboard_file.write(page)
    return dashboard_path, metadata_path


def main():
    args = _parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Forecast output directory does not exist: {output_dir}")

    metadata_path = os.path.join(output_dir, "forecast_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if args.forecast_init:
            metadata["forecast_initialization_utc"] = _parse_utc(
                args.forecast_init
            ).isoformat(timespec="seconds")
        print(f"[DASHBOARD] Loaded saved forecast metadata: {metadata_path}")
    else:
        metadata = _infer_metadata(output_dir, args.forecast_init)
        print(
            "[DASHBOARD] No saved metadata found. Recovered available time/grid "
            "details from the existing forecast NetCDF; original model/run "
            "settings will be shown as not recorded."
        )

    extra_files = [
        os.path.abspath(path)
        for path in (args.input_grib, args.weights)
        if path
    ]
    dashboard_path, metadata_path = write_forecast_dashboard(
        output_dir=output_dir,
        metadata=metadata,
        inventory_root=args.inventory_root,
        extra_files=extra_files,
    )
    print(f"[DASHBOARD] HTML: {dashboard_path}")
    print(f"[DASHBOARD] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
