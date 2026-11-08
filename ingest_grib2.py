# ingest_grib2.py
import glob
import os
import threading
import time
import warnings
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")

SURFACE_VARS = {
    "t2m": ["t2m", "2t"],
    "d2m": ["d2m", "2d", "sh2"],
    "u10": ["u10", "10u"],
    "v10": ["v10", "10v"],
    "msl": ["msl", "prmsl", "sp"],
    "tp":  ["tp", "acpcp"]
}

STATIC_VARS = {
    "orog": ["orog", "z", "hgt"],
    "lsm":  ["lsm", "land"]
}

UPPER_AIR_VARS = {
    "z": ["z", "gh"],
    "t": ["t"],
    "q": ["q", "r"],
    "u": ["u"],
    "v": ["v"]
}

REQUIRED_PLEVS = [850, 700, 500, 250]
UPPER_AIR_KEYS = tuple(
    f"{std_name}_{plev}" for std_name in UPPER_AIR_VARS for plev in REQUIRED_PLEVS
)
ERA5_LAND_VARS = {
    "land_skin_temp": ["skt", "skin_temperature"],
    "soil_temp_1": ["stl1", "soil_temperature_level_1"],
    "soil_moisture_1": ["swvl1", "volumetric_soil_water_layer_1"],
    "snow_cover": ["snowc", "snow_cover"],
}
ERA5_LAND_KEYS = tuple(ERA5_LAND_VARS)
ERA5_LAND_ALIASES = tuple(frozenset(aliases) for aliases in ERA5_LAND_VARS.values())
GRID_SEQUENCE_KEYS = (
    tuple(SURFACE_VARS)
    + tuple(
        f"{variable}_{pressure_level}"
        for variable in UPPER_AIR_VARS
        for pressure_level in REQUIRED_PLEVS
    )
    + tuple(STATIC_VARS)
    + ERA5_LAND_KEYS
)


def grib_has_era5_land_fields(grib_path):
    started = time.monotonic()
    scanned_messages = 0
    available_fields = set()
    try:
        import eccodes

        print(
            f"[CDS] Scanning GRIB message headers for ERA5-Land fields: "
            f"{grib_path}",
            flush=True,
        )
        with open(grib_path, "rb") as grib_file:
            last_progress = started
            while True:
                message = eccodes.codes_grib_new_from_file(grib_file)
                if message is None:
                    break
                try:
                    available_fields.add(eccodes.codes_get(message, "shortName"))
                    scanned_messages += 1
                finally:
                    eccodes.codes_release(message)

                now = time.monotonic()
                if now - last_progress >= 30:
                    print(
                        f"[CDS] Scanned {scanned_messages} GRIB messages "
                        f"({now - started:.0f}s elapsed): {grib_path}",
                        flush=True,
                    )
                    last_progress = now

        if scanned_messages == 0:
            print(f"[WARN] No readable GRIB messages found in {grib_path}.", flush=True)
            return False

        missing_fields = [
            aliases for aliases in ERA5_LAND_ALIASES
            if not available_fields.intersection(aliases)
        ]
        if missing_fields:
            missing_names = ", ".join("/".join(sorted(names)) for names in missing_fields)
            print(
                f"[CDS] Required ERA5-Land GRIB fields not found: {missing_names} "
                f"(scanned {scanned_messages} messages in "
                f"{time.monotonic() - started:.1f}s)",
                flush=True,
            )
            return False
        print(
            f"[CDS] ERA5-Land field verification completed: found all required "
            f"fields in {scanned_messages} messages "
            f"({time.monotonic() - started:.1f}s): {grib_path}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(
            f"[WARN] Could not scan GRIB headers in {grib_path} after "
            f"{time.monotonic() - started:.1f}s and {scanned_messages} messages: "
            f"{exc}",
            flush=True,
        )
        return False


def _extract_var(ds_list, alias_list, exclude_dims=(), valid_time=None):
    for ds in ds_list:
        for alias in alias_list:
            if alias in ds.data_vars:
                da = ds[alias]
                if (
                    not any(dim in da.dims for dim in exclude_dims)
                    and _has_valid_time(da, valid_time)
                ):
                    return da
    return None


def _has_valid_time(da, valid_time):
    if valid_time is None:
        return True
    found_time_coordinate = False
    for coord_name in ("valid_time", "time"):
        if coord_name in da.coords:
            found_time_coordinate = True
            values = np.asarray(da.coords[coord_name].values).astype("datetime64[ns]")
            if np.any(values == np.datetime64(valid_time, "ns")):
                return True
    return not found_time_coordinate and "time" not in da.dims


def _select_valid_time(da, valid_time):
    if valid_time is None:
        if "time" in da.dims:
            return da.isel(time=-1)
        return da

    for coord_name in ("valid_time", "time"):
        if coord_name in da.coords:
            coord = da.coords[coord_name]
            values = np.asarray(coord.values).astype("datetime64[ns]")
            matches = np.flatnonzero(values.reshape(-1) == np.datetime64(valid_time, "ns"))
            if matches.size:
                indices = np.unravel_index(matches[0], values.shape)
                indexers = {
                    dim: int(index)
                    for dim, index in zip(coord.dims, indices)
                    if dim in da.dims
                }
                return da.isel(indexers)
    if "time" in da.dims:
        raise ValueError(f"GRIB field does not contain requested valid time {valid_time}")
    return da


def _regrid_field(da, lat_name, lon_name, target_lats, target_lons, var_name, valid_time=None):
    da = _select_valid_time(da, valid_time)

    for dim in [dim for dim in da.dims if dim not in (lat_name, lon_name)]:
        if da.sizes[dim] != 1:
            raise ValueError(
                f"Cannot regrid {var_name}: unexpected non-spatial dimension "
                f"{dim!r} has size {da.sizes[dim]}"
            )
        da = da.isel({dim: 0}, drop=True)

    da_interp = da.interp({lat_name: target_lats, lon_name: target_lons}, method="linear")
    values = np.array(da_interp.transpose(lat_name, lon_name).values, copy=True)
    expected_shape = (len(target_lats), len(target_lons))
    if values.shape != expected_shape:
        raise ValueError(
            f"Cannot regrid {var_name}: expected shape {expected_shape}, got {values.shape}"
        )
    return _fill_missing_values(values, var_name)


def _fill_missing_values(values, var_name):
    missing = ~np.isfinite(values)
    if np.any(missing):
        valid = ~missing
        if not np.any(valid):
            raise ValueError(f"Cannot regrid {var_name}: field contains no finite values")
        _, nearest_indices = distance_transform_edt(missing, return_indices=True)
        values[missing] = values[tuple(index[missing] for index in nearest_indices)]
        print(
            f"[WARN] Filled {np.count_nonzero(missing)} missing {var_name} grid cells "
            "from their nearest valid cells."
        )
    return values


def _regrid_sequence_field(
    da, lat_name, lon_name, target_lats, target_lons, var_name, valid_times
):
    temporal_coord = None
    for coord_name in ("valid_time", "time"):
        if coord_name in da.coords:
            coord = da.coords[coord_name]
            if coord.dims and all(dim in da.dims for dim in coord.dims):
                temporal_coord = coord
                break

    da_interp = da.interp(
        {lat_name: target_lats, lon_name: target_lons}, method="linear"
    )
    temporal_dims = () if temporal_coord is None else temporal_coord.dims
    for dim in [
        dim for dim in da_interp.dims
        if dim not in (lat_name, lon_name) and dim not in temporal_dims
    ]:
        if da_interp.sizes[dim] != 1:
            raise ValueError(
                f"Cannot regrid {var_name}: unexpected non-spatial dimension "
                f"{dim!r} has size {da_interp.sizes[dim]}"
            )
        da_interp = da_interp.isel({dim: 0}, drop=True)

    expected_shape = (len(target_lats), len(target_lons))
    if temporal_coord is None:
        values = np.asarray(
            da_interp.transpose(lat_name, lon_name).values, dtype=np.float32
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"Cannot regrid {var_name}: expected shape {expected_shape}, "
                f"got {values.shape}"
            )
        values = _fill_missing_values(values, var_name)
        return {valid_time: values for valid_time in valid_times}

    times = np.asarray(temporal_coord.values).astype("datetime64[ns]")
    values = np.asarray(
        da_interp.transpose(*temporal_dims, lat_name, lon_name).values,
        dtype=np.float32,
    )
    if times.size != int(np.prod(values.shape[:-2])):
        raise ValueError(
            f"Cannot regrid {var_name}: time coordinate shape {times.shape} "
            f"does not match data dimensions {values.shape}"
        )
    times = times.reshape(-1)
    values = values.reshape((-1,) + expected_shape)
    requested_times = set(valid_times)
    regridded = {}
    for timestamp, field in zip(times, values):
        if np.isnat(timestamp) or timestamp not in requested_times:
            continue
        regridded[timestamp] = _fill_missing_values(field, var_name)
    return regridded


def _available_valid_times(ds_list):
    for aliases in SURFACE_VARS.values():
        da = _extract_var(ds_list, aliases)
        if da is None:
            continue
        for coord_name in ("valid_time", "time"):
            if coord_name in da.coords:
                values = np.asarray(da.coords[coord_name].values).astype("datetime64[ns]")
                values = values[~np.isnat(values)]
                return np.unique(values.reshape(-1))
    return np.array([], dtype="datetime64[ns]")


def _regrid_datasets(ds_list, bbox, target_res, valid_time=None):
    target_lats = np.arange(bbox["lat_min"], bbox["lat_max"] + target_res / 2, target_res)
    target_lons = np.arange(bbox["lon_min"], bbox["lon_max"] + target_res / 2, target_res)

    sample_ds = ds_list[0]
    lat_name = "latitude" if "latitude" in sample_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in sample_ds.coords else "lon"

    if sample_ds[lon_name].max() > 180 and bbox["lon_min"] < 0:
        for i in range(len(ds_list)):
            ds_list[i] = ds_list[i].assign_coords(
                {lon_name: (((ds_list[i][lon_name] + 180) % 360) - 180)}
            ).sortby(lon_name)

    regridded_data = {}

    for std_name, aliases in SURFACE_VARS.items():
        da = _extract_var(ds_list, aliases, valid_time=valid_time)
        if da is not None:
            regridded_data[std_name] = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name, valid_time
            )
        else:
            if valid_time is not None:
                raise ValueError(
                    f"Historical GRIB frame {valid_time} is missing required field {std_name}"
                )
            regridded_data[std_name] = _default_field(std_name, len(target_lats), len(target_lons))

    for std_name, aliases in STATIC_VARS.items():
        da = _extract_var(
            ds_list,
            aliases,
            exclude_dims=("isobaricInhPa",),
            valid_time=valid_time,
        )
        if da is not None:
            vals = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name, valid_time
            )
            if std_name == "orog" and np.nanmax(vals) > 9000:
                vals = vals / 9.80665
            regridded_data[std_name] = vals
        else:
            regridded_data[std_name] = _default_field(std_name, len(target_lats), len(target_lons))

    for std_name, aliases in ERA5_LAND_VARS.items():
        da = _extract_var(ds_list, aliases, valid_time=valid_time)
        if da is not None:
            regridded_data[std_name] = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name, valid_time
            )
        else:
            if valid_time is not None:
                raise ValueError(
                    f"Historical GRIB frame {valid_time} is missing ERA5-Land field {std_name}"
                )
            regridded_data[std_name] = _default_field(
                std_name, len(target_lats), len(target_lons)
            )

    for std_name, aliases in UPPER_AIR_VARS.items():
        da = _extract_var(ds_list, aliases, valid_time=valid_time)
        for plev in REQUIRED_PLEVS:
            key_name = f"{std_name}_{plev}"
            if da is not None and "isobaricInhPa" in da.coords:
                if plev in da["isobaricInhPa"].values:
                    da_plev = da.sel(isobaricInhPa=plev)
                    regridded_data[key_name] = _regrid_field(
                        da_plev,
                        lat_name,
                        lon_name,
                        target_lats,
                        target_lons,
                        key_name,
                        valid_time,
                    )
                    continue
            regridded_data[key_name] = _default_field(key_name, len(target_lats), len(target_lons))

    return regridded_data, target_lats, target_lons


def _regrid_datasets_sequence(ds_list, bbox, target_res, valid_times):
    target_lats = np.arange(
        bbox["lat_min"], bbox["lat_max"] + target_res / 2, target_res
    )
    target_lons = np.arange(
        bbox["lon_min"], bbox["lon_max"] + target_res / 2, target_res
    )
    sample_ds = ds_list[0]
    lat_name = "latitude" if "latitude" in sample_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in sample_ds.coords else "lon"

    if sample_ds[lon_name].max() > 180 and bbox["lon_min"] < 0:
        for index, dataset in enumerate(ds_list):
            ds_list[index] = dataset.assign_coords(
                {lon_name: (((dataset[lon_name] + 180) % 360) - 180)}
            ).sortby(lon_name)

    fields = {}

    def add_field(key, aliases, exclude_dims=(), pressure_level=None):
        da = _extract_var(ds_list, aliases, exclude_dims=exclude_dims)
        if da is None:
            return
        if pressure_level is not None:
            levels = da.coords.get("isobaricInhPa")
            if levels is None or pressure_level not in levels.values:
                return
            da = da.sel(isobaricInhPa=pressure_level)
        values = _regrid_sequence_field(
            da, lat_name, lon_name, target_lats, target_lons, key, valid_times
        )
        if key in STATIC_VARS and values:
            static_field = next(iter(values.values()))
            values = {timestamp: static_field for timestamp in valid_times}
        if key == "orog":
            values = {
                timestamp: (
                    field / 9.80665 if np.nanmax(field) > 9000 else field
                )
                for timestamp, field in values.items()
            }
        fields[key] = values

    print("[INGEST] Regridding ERA5 surface fields...")
    for key, aliases in SURFACE_VARS.items():
        add_field(key, aliases)
    print("[INGEST] Regridding static terrain and land-sea fields...")
    for key, aliases in STATIC_VARS.items():
        add_field(key, aliases, exclude_dims=("isobaricInhPa",))
    print("[INGEST] Regridding ERA5-Land fields...")
    for key, aliases in ERA5_LAND_VARS.items():
        add_field(key, aliases)
    print("[INGEST] Regridding upper-air pressure-level fields...")
    for variable, aliases in UPPER_AIR_VARS.items():
        for pressure_level in REQUIRED_PLEVS:
            key = f"{variable}_{pressure_level}"
            add_field(key, aliases, pressure_level=pressure_level)

    required_keys = set(SURFACE_VARS)
    missing_land_fields = set(ERA5_LAND_VARS).difference(fields)
    if missing_land_fields:
        print(
            "[WARN] Historical archive is missing ERA5-Land field(s): "
            + ", ".join(sorted(missing_land_fields))
            + "; using neutral defaults for those input channels.",
            flush=True,
        )
    frames = []
    for timestamp in valid_times:
        grid_data = {}
        for key in GRID_SEQUENCE_KEYS:
            field = fields.get(key, {}).get(timestamp)
            if field is None:
                if key in required_keys:
                    category = (
                        "ERA5-Land" if key in ERA5_LAND_VARS else "required"
                    )
                    raise ValueError(
                        f"Historical GRIB frame {timestamp} is missing "
                        f"{category} field {key}"
                    )
                field = _default_field(key, len(target_lats), len(target_lons))
            grid_data[key] = field
        frame_time = timestamp.astype("datetime64[us]").astype(object)
        frames.append((frame_time, grid_data))
    return frames, target_lats, target_lons


def _open_grib_datasets(grib_path):
    import cfgrib

    started = time.monotonic()
    finished = threading.Event()
    empty_indexes = [
        index_path
        for index_path in glob.glob(grib_path + ".*.idx")
        if os.path.getsize(index_path) == 0
    ]
    index_generation = 0
    while any(
        os.path.getsize(index_path) == 0
        for index_path in glob.glob(
            f"{grib_path}.cfgrib-cache-{index_generation}-*.idx"
        )
    ):
        index_generation += 1
    recovery_indexpath = (
        f"{grib_path}.cfgrib-cache-{index_generation}-{{short_hash}}.idx"
    )
    use_default_index_cache = not empty_indexes
    if empty_indexes:
        print(
            "[GRIB] Empty cfgrib index sidecar(s) found; ignoring them and "
            f"using a separate persistent index: {', '.join(empty_indexes)}",
            flush=True,
        )

    def report_progress():
        while not finished.wait(30):
            elapsed = time.monotonic() - started
            print(
                f"[GRIB] Still scanning {grib_path} "
                f"({elapsed:.0f}s elapsed; index cache "
                f"{'enabled' if use_default_index_cache else 'rebuilding'}).",
                flush=True,
            )

    print(
        f"[GRIB] Starting cfgrib scan of {grib_path} "
        f"(index cache {'enabled' if use_default_index_cache else 'rebuilding'}).",
        flush=True,
    )
    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()
    try:
        if use_default_index_cache:
            try:
                datasets = cfgrib.open_datasets(grib_path)
            except Exception as cached_error:
                print(
                    f"[GRIB] Cached-index open failed for {grib_path}: "
                    f"{cached_error}. Retrying with a separate persistent index.",
                    flush=True,
                )
                datasets = cfgrib.open_datasets(
                    grib_path,
                    backend_kwargs={"indexpath": recovery_indexpath},
                )
        else:
            datasets = cfgrib.open_datasets(
                grib_path, backend_kwargs={"indexpath": recovery_indexpath}
            )
    except Exception as exc:
        print(
            f"[GRIB] Scan failed after {time.monotonic() - started:.1f}s: {exc}",
            flush=True,
        )
        raise
    finally:
        finished.set()
        reporter.join()
    print(
        f"[GRIB] cfgrib opened {len(datasets)} dataset group(s) in "
        f"{time.monotonic() - started:.1f}s: {grib_path}",
        flush=True,
    )
    return datasets


def load_and_regrid_grib2_sequence(grib_path, bbox, target_res):
    print(f"[INGEST] Loading GRIB2 time sequence: {grib_path}")
    try:
        ds_list = _open_grib_datasets(grib_path)
    except Exception as e:
        print(f"[WARN] Could not open {grib_path} ({e}). Using one synthetic frame.")
        grid_data, lats, lons = generate_synthetic_grid(bbox, target_res)
        return [(None, grid_data)], lats, lons

    try:
        valid_times = _available_valid_times(ds_list)
        if not valid_times.size:
            grid_data, lats, lons = _regrid_datasets(ds_list, bbox, target_res)
            return [(None, grid_data)], lats, lons

        print(
            f"[INGEST] Regridding {len(valid_times)} timestamps in batches "
            "to avoid repeating interpolation for every frame."
        )
        frames, lats, lons = _regrid_datasets_sequence(
            ds_list, bbox, target_res, valid_times
        )
        print(f"[INGEST] Regridded {len(frames)} frames from {grib_path}.")
        return frames, lats, lons
    finally:
        for dataset in ds_list:
            dataset.close()


def load_and_regrid_grib2(grib_path, bbox, target_res):
    """
    Ingests GRIB2, crops to bounding box, and interpolates to target resolution.
    Falls back to synthetic grid if the file is missing or eccodes fails.
    """
    print(f"[INGEST] Loading GRIB2 file: {grib_path}")
    try:
        ds_list = _open_grib_datasets(grib_path)
    except Exception as e:
        print(f"[WARN] Could not open {grib_path} ({e}). Generating synthetic operational grid for standalone testing...")
        return generate_synthetic_grid(bbox, target_res)

    try:
        return _regrid_datasets(ds_list, bbox, target_res)
    finally:
        for dataset in ds_list:
            dataset.close()


def _default_field(var_name, nlat, nlon):
    defaults = {
        "t2m": 288.15,
        "d2m": 280.15,
        "u10": 3.0,
        "v10": 1.5,
        "msl": 101325.0,
        "tp": 0.0,
        "orog": 150.0,
        "lsm": 0.5,
        "land_skin_temp": 288.15,
        "soil_temp_1": 288.15,
        "soil_moisture_1": 0.25,
        "snow_cover": 0.0,
    }
    return np.full((nlat, nlon), defaults.get(var_name, 0.0), dtype=np.float32)


def generate_synthetic_grid(bbox, target_res):
    target_lats = np.arange(bbox["lat_min"], bbox["lat_max"] + target_res / 2, target_res)
    target_lons = np.arange(bbox["lon_min"], bbox["lon_max"] + target_res / 2, target_res)
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)

    orog = np.clip(600 * np.sin((lon_grid - bbox["lon_min"]) * 0.5) * np.cos((lat_grid - bbox["lat_min"]) * 0.3), 0, 2500)
    lsm = (orog > 10).astype(np.float32)

    t2m = 295.0 - 0.6 * (lat_grid - bbox["lat_min"]) - 0.0065 * orog + 2.5 * np.sin(lon_grid * 0.4)
    d2m = t2m - (4.0 + 5.0 * lsm)
    msl = 101325.0 - 800.0 * np.exp(-((lon_grid - np.mean(target_lons))**2 + (lat_grid - np.mean(target_lats))**2) / 15.0)
    u10 = 6.0 + 3.0 * np.cos(lat_grid * 0.2) * (1.0 - 0.3 * lsm)
    v10 = 2.5 * np.sin(lon_grid * 0.3)
    tp = np.clip(
        2.0
        * np.exp(
            -((lon_grid - np.mean(target_lons))**2 + (lat_grid - np.mean(target_lats))**2)
            / 8.0
        ),
        0,
        25,
    ) / 1000.0

    data = {
        "t2m": t2m,
        "d2m": d2m,
        "u10": u10,
        "v10": v10,
        "msl": msl,
        "tp": tp,
        "orog": orog,
        "lsm": lsm,
        "land_skin_temp": t2m.copy(),
        "soil_temp_1": t2m - 1.5,
        "soil_moisture_1": np.full_like(t2m, 0.25),
        "snow_cover": np.zeros_like(t2m),
    }
    for std_name in UPPER_AIR_VARS:
        for plev in REQUIRED_PLEVS:
            data[f"{std_name}_{plev}"] = np.full_like(t2m, 10.0 if std_name == "u" else 260.0)
    return data, target_lats, target_lons
