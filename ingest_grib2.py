# ingest_grib2.py
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


def grib_has_era5_land_fields(grib_path):
    import cfgrib

    datasets = cfgrib.open_datasets(grib_path)
    try:
        available_vars = set().union(*(dataset.data_vars for dataset in datasets))
        return all(available_vars.intersection(aliases) for aliases in ERA5_LAND_ALIASES)
    finally:
        for dataset in datasets:
            dataset.close()


def _extract_var(ds_list, alias_list, exclude_dims=()):
    for ds in ds_list:
        for alias in alias_list:
            if alias in ds.data_vars:
                da = ds[alias]
                if not any(dim in da.dims for dim in exclude_dims):
                    return da
    return None


def _regrid_field(da, lat_name, lon_name, target_lats, target_lons, var_name):
    if "time" in da.dims:
        da = da.isel(time=-1)

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


def load_and_regrid_grib2(grib_path, bbox, target_res):
    """
    Ingests GRIB2, crops to bounding box, and interpolates to target resolution.
    Falls back to synthetic grid if the file is missing or eccodes fails.
    """
    print(f"[INGEST] Loading GRIB2 file: {grib_path}")
    try:
        import cfgrib
        ds_list = cfgrib.open_datasets(grib_path)
    except Exception as e:
        print(f"[WARN] Could not open {grib_path} ({e}). Generating synthetic operational grid for standalone testing...")
        return generate_synthetic_grid(bbox, target_res)

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
        da = _extract_var(ds_list, aliases)
        if da is not None:
            regridded_data[std_name] = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name
            )
        else:
            regridded_data[std_name] = _default_field(std_name, len(target_lats), len(target_lons))

    for std_name, aliases in STATIC_VARS.items():
        da = _extract_var(ds_list, aliases, exclude_dims=("isobaricInhPa",))
        if da is not None:
            vals = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name
            )
            if std_name == "orog" and np.nanmax(vals) > 9000:
                vals = vals / 9.80665
            regridded_data[std_name] = vals
        else:
            regridded_data[std_name] = _default_field(std_name, len(target_lats), len(target_lons))

    for std_name, aliases in ERA5_LAND_VARS.items():
        da = _extract_var(ds_list, aliases)
        if da is not None:
            regridded_data[std_name] = _regrid_field(
                da, lat_name, lon_name, target_lats, target_lons, std_name
            )
        else:
            regridded_data[std_name] = _default_field(
                std_name, len(target_lats), len(target_lons)
            )

    for std_name, aliases in UPPER_AIR_VARS.items():
        da = _extract_var(ds_list, aliases)
        for plev in REQUIRED_PLEVS:
            key_name = f"{std_name}_{plev}"
            if da is not None and "isobaricInhPa" in da.coords:
                if plev in da["isobaricInhPa"].values:
                    da_plev = da.sel(isobaricInhPa=plev)
                    regridded_data[key_name] = _regrid_field(
                        da_plev, lat_name, lon_name, target_lats, target_lons, key_name
                    )
                    continue
            regridded_data[key_name] = _default_field(key_name, len(target_lats), len(target_lons))

    return regridded_data, target_lats, target_lons


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
    tp = np.clip(2.0 * np.exp(-((lon_grid - np.mean(target_lons))**2 + (lat_grid - np.mean(target_lats))**2) / 8.0), 0, 25)

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
