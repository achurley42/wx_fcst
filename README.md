# wx_fcst
72-hr regional weather forecasting scripts written in python outputting NetCDF4 cubes, point-forecast CSV meteograms, and multi-panel prognostic charts.

# Standalone Regional AI Weather Forecasting System (72-Hour Rollout)

An operational, modular Python forecasting pipeline designed for standalone meteorology and data analysis workstations. The system ingests **GRIB2** background analysis grids (surface, isobaric upper-air, and static topography/land-sea masks), dynamically subsets and regrids to user-specified bounding boxes and resolutions, and executes an autoregressive 72-hour forecast (at 6-hour intervals) outputting **NetCDF4 cubes, point-forecast CSV meteograms, and multi-panel prognostic charts**.

The pipeline features a **Physics-Guided Neural Hybrid Architecture**: it can execute out of the box with zero training using physical advection and diurnal constraints, or leverage a fine-tuned PyTorch Spatio-Temporal Residual U-Net (`.pt` checkpoint).

---

## 📋 Table of Contents
1. [System Architecture & Capabilities](#system-architecture--capabilities)
2. [Repository Structure](#repository-structure)
3. [Environment Setup & Installation](#environment-setup--installation)
4. [Input Data Requirements & GRIB2 Specifications](#input-data-requirements--grib2-specifications)
5. [Operational Execution (Inference)](#operational-execution-inference)
   - [Zero-Training Mode (Physics-Guided Hybrid)](#zero-training-mode-physics-guided-hybrid)
   - [Fine-Tuned Model Inference](#fine-tuned-model-inference)
   - [Synthetic Verification Run](#synthetic-verification-run)
6. [Model Training & Fine-Tuning](#model-training--fine-tuning)
7. [Output Deliverables](#output-deliverables)
8. [Hardware & Resource Management](#hardware--resource-management)
9. [Troubleshooting & Operational Notes](#troubleshooting--operational-notes)

---

## ⚙️ System Architecture & Capabilities

```text
+--------------------------------------------------------------------------+
|                            INPUT DATA LAYER                              |
|   ERA5 / GALWEM / GFS Blended Analysis (GRIB2) or Synthetic Generator   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  ingest_grib2.py (Ingestion & Regridding)                |
|  - Dynamically subsets to user Bounding Box [lat_min, lat_max, etc.]     |
|  - Bi-linear interpolation to user-defined resolution (e.g., 0.05 deg)   |
|  - Static field extraction: Orography (orog/z) & Land-Sea Mask (lsm)     |
|  - Upper-air isobaric extraction (850, 700, 500, 250 hPa)                |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|               model_architecture.py (Compute & Core Model)               |
|  - HybridComputeManager: CUDA GPU -> Apple Silicon MPS -> Multi-Core CPU |
|  - Automatic fallback to CPU if GPU encounters Out-Of-Memory (OOM)       |
|  - WeatherSpatioTemporalUNet: 30-channel input -> 6-channel residual     |
+------------------------------------+-------------------------------------+
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
+------------------------------------+   +---------------------------------+
| forecast_engine.py (Rollout)       |   | train_unet.py (Training)        |
| - 12-step autoregressive rollout   |   | - 6-hourly paired GRIB2 loader  |
| - 850 hPa semi-Lagrangian advection|   | - Latitude-weighted Huber loss  |
| - Solar zenith diurnal forcing     |   | - Saturation penalty ($T_d \le  |
| - LSM differential heat capacity   |   |   T$)                           |
| - Thermodynamic limits ($T_d \le   |   | - FP16 mixed precision scaling  |
|   T$)                              |   +---------------------------------+
+-----------------+------------------+
                  |
                  v
+--------------------------------------------------------------------------+
|                           OUTPUT DELIVERABLES                            |
|  1. forecast_72hr_surface.nc     : Complete 4D NetCDF CF-compliant cube  |
|  2. aor_center_point_forecast.csv: Operational meteogram tabular summary |
|  3. surface_prog_T+XXh.png       : 4-Panel diagnostic surface prog charts|
+--------------------------------------------------------------------------+
```

---

## 📂 Repository Structure

```text
├── environment.yml             # Conda environment definition
├── ingest_grib2.py             # GRIB2 ingestion, subsetting, regridding, synthetic generator
├── model_architecture.py       # Hybrid compute manager and PyTorch U-Net architecture
├── forecast_engine.py          # Autoregressive rollout engine, physics hooks, deliverable exporters
├── run_forecast.py             # Operational CLI entrypoint for running forecasts
├── train_unet.py               # Model training and fine-tuning script
├── historical_grib2/           # Directory for multi-year GRIB2 archives for training (user-provided)
├── checkpoints/                # Saved model weights (.pt files)
└── forecast_output/            # Generated NetCDF, CSV, and PNG forecast products
```

---

## 🛠️ Environment Setup & Installation

The pipeline relies on standard scientific Python packages alongside compiled libraries for handling WMO GRIB2 formats (`eccodes` and `cfgrib`).

### Option 1: Fast Setup via Conda (Recommended)

Create and activate the environment using `conda-forge`:

```bash
conda create -n af_ai_wx -c conda-forge python=3.11 \
    cfgrib \
    eccodes \
    xarray \
    netcdf4 \
    scipy \
    numpy \
    pandas \
    matplotlib \
    cartopy \
    metpy \
    pytorch \
    torchvision \
    -y

conda activate af_ai_wx
```

> **GPU Note:** If your workstation possesses an NVIDIA GPU with CUDA drivers installed, install PyTorch with CUDA support instead of standard CPU packages:
> ```bash
> conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1
> ```

### Option 2: Using the Included `environment.yml`

```yaml
name: af_ai_wx
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch
  - torchvision
  - cfgrib
  - eccodes
  - xarray
  - netcdf4
  - scipy
  - numpy
  - pandas
  - matplotlib
  - cartopy
  - metpy
```

Run:
```bash
conda env create -f environment.yml
conda activate af_ai_wx
```

---

## 🛰️ Input Data Requirements & GRIB2 Specifications

The ingestion module (`ingest_grib2.py`) is designed to parse standard **ERA5**, **NCEP GFS/GDAS**, or **Air Force Blended Analysis** GRIB2 datasets.

### Required Variable Hypercubes

1. **Surface Parameters (`SURFACE_VARS`):**
   - 2m Temperature: `t2m` or `2t` (Kelvin)
   - 2m Dewpoint Temperature: `d2m`, `2d`, or `sh2` (Kelvin)
   - 10m U-component of Wind: `u10` or `10u` ($\text{m/s}$)
   - 10m V-component of Wind: `v10` or `10v` ($\text{m/s}$)
   - Mean Sea Level Pressure: `msl`, `prmsl`, or `sp` ($\text{Pa}$)
   - Total Precipitation / Convective: `tp` or `acpcp` ($\text{m}$ or $\text{kg/m}^2$)

2. **Static Surface Boundary Conditions (`STATIC_VARS`):**
   - Surface Orography / Elevation: `orog`, `z`, or `hgt` ($\text{m}$ or $\text{m}^2/\text{s}^2$)
   - Land-Sea Mask: `lsm` or `land` (dimensionless fraction, $0.0 = \text{water}$, $1.0 = \text{land}$)

3. **Upper-Air Isobaric Levels (`UPPER_AIR_VARS` at 850, 700, 500, 250 hPa):**
   - Geopotential Height: `z` or `gh` ($\text{m}^2/\text{s}^2$ or $\text{gpm}$)
   - Temperature: `t` (Kelvin)
   - Specific or Relative Humidity: `q` or `r`
   - Horizontal Winds: `u` and `v` ($\text{m/s}$)

*Note: If an isobaric level or static parameter is missing from the input file, the script applies an automated physical baseline default rather than halting execution.*

---

## 🚀 Operational Execution (Inference)

Execute forecasts using `run_forecast.py`. All coordinate boundaries and spatial resolutions are passed via CLI flags.

### 1. Zero-Training Mode (Physics-Guided Hybrid)
You do **not** need a trained weight file to produce an operational forecast. Running without `--weights` activates the physical steering, diurnal heating, and lapse-rate formulations directly:

```bash
python run_forecast.py \
  --grib ./analysis_current.grib2 \
  --lat-min 24.0 \
  --lat-max 35.0 \
  --lon-min -90.0 \
  --lon-max -75.0 \
  --res 0.05 \
  --outdir ./forecast_output
```

### 2. Fine-Tuned Model Inference
If you have trained weights or an official checkpoint:

```bash
python run_forecast.py \
  --grib ./analysis_current.grib2 \
  --lat-min 24.0 \
  --lat-max 35.0 \
  --lon-min -90.0 \
  --lon-max -75.0 \
  --res 0.05 \
  --weights ./checkpoints/regional_unet_best.pt \
  --outdir ./forecast_output
```

### 3. Synthetic Verification Run (No GRIB2 File Required)
To verify your workstation configuration, GPU detection, and rendering stack without waiting for external data transfers, omit the `--grib` argument or point to a non-existent file. The system will automatically synthesize a baroclinic wave, coastal boundary, and mountain range matching your bounding box:

```bash
python run_forecast.py \
  --lat-min 30.0 \
  --lat-max 42.0 \
  --lon-min -125.0 \
  --lon-max -110.0 \
  --res 0.10 \
  --outdir ./test_synthetic_run
```

### Command-Line Arguments Reference (`run_forecast.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grib` | `str` | `input_blended.grib2` | Path to the input GRIB2 initialization file |
| `--lat-min` | `float` | *Required* | Southern boundary latitude (e.g. `25.0`) |
| `--lat-max` | `float` | *Required* | Northern boundary latitude (e.g. `35.0`) |
| `--lon-min` | `float` | *Required* | Western boundary longitude in degrees East/West (`-180` to `180`) |
| `--lon-max` | `float` | *Required* | Eastern boundary longitude in degrees East/West (`-180` to `180`) |
| `--res` | `float` | `0.10` | Output spatial resolution in degrees (`0.10` $\approx$ 11 km, `0.05` $\approx$ 5.5 km) |
| `--weights` | `str` | `None` | Path to trained `.pt` PyTorch weight checkpoint |
| `--force-cpu` | `flag` | `False` | Forces CPU execution regardless of detected GPUs |
| `--outdir` | `str` | `./forecast_output` | Destination directory for generated products |

---

## 🏋️ Model Training & Fine-Tuning

To train the U-Net on historical data, organize chronological GRIB2 files at 6-hour intervals in a target directory:

```text
historical_grib2/
  ├── era5_20240101_00z.grib2
  ├── era5_20240101_06z.grib2
  ├── era5_20240101_12z.grib2
  └── era5_20240101_18z.grib2
```

Run `train_unet.py`:

```bash
python train_unet.py \
  --grib-dir ./historical_grib2 \
  --lat-min 24.0 \
  --lat-max 35.0 \
  --lon-min -90.0 \
  --lon-max -75.0 \
  --res 0.05 \
  --epochs 25 \
  --batch-size 4 \
  --lr 0.001 \
  --save-weights ./checkpoints/theater_forecast_model.pt
```

### Physics-Informed Loss Function
The training script employs `PhysicsInformedWeatherLoss`, combining:
1. **Latitude-Weighted Huber Loss:** Weighs grid error by $\cos(\text{latitude})$ to maintain spatial conservation across converging meridians.
2. **Thermodynamic Saturation Penalty:** Explicitly calculates $T_{2\text{m}}$ and $T_{d,2\text{m}}$, applying quadratic penalties whenever $T_d > T$ (enforcing $\text{RH} \le 100\%$).

*Note: If `./historical_grib2` contains fewer than 2 files, `train_unet.py` automatically generates a synthetic historical sequence to validate the training pipeline.*

---

## 📊 Output Deliverables

Upon completion of the 12-step rollout ($T+00\text{h}$ to $T+72\text{h}$), the following files are produced in the designated output folder:

### 1. `forecast_72hr_surface.nc` (NetCDF4 Grid Cube)
Contains 4D/3D spatial variables indexed across `[time, lat, lon]`:
- `t2m`: 2m Temperature ($^\circ\text{C}$)
- `d2m`: 2m Dewpoint ($^\circ\text{C}$)
- `u10`, `v10`: Surface Wind Components ($\text{m/s}$)
- `msl`: Mean Sea Level Pressure ($\text{hPa}$)
- `tp`: 6-hour Accumulated Precipitation ($\text{mm}$)
- `orog`: Topographic Orography ($\text{m}$)
- `lsm`: Land-Sea Mask ($0.0 - 1.0$)

### 2. `aor_center_point_forecast.csv` (Tabular Meteogram)
A meteogram summary formatted for direct briefing use:

| Forecast_Hour | Valid_Time_UTC | Center_Lat | Center_Lon | Temp_2m_C | Dewpoint_2m_C | Wind_Dir_Deg | Wind_Spd_Kts | MSLP_hPa | Precip_6hr_mm |
|---|---|---|---|---|---|---|---|---|---|
| T+00h | 2026-03-30 00:00Z | 29.5 | -82.5 | 18.2 | 14.1 | 045 | 11.2 | 1016.4 | 0.00 |
| T+06h | 2026-03-30 06:00Z | 29.5 | -82.5 | 15.4 | 13.8 | 060 | 9.5 | 1015.8 | 0.45 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| T+72h | 2026-04-02 00:00Z | 29.5 | -82.5 | 21.0 | 16.5 | 180 | 14.0 | 1008.2 | 6.20 |

### 3. Surface Prognostic Charts (`surface_prog_T+XXh.png`)
Rendered at **$T+24\text{h}$**, **$T+48\text{h}$**, and **$T+72\text{h}$**. Each graphic contains 4 panels:
- **Upper-Left:** 2m Temperature ($^\circ\text{C}$) with coastlines and terrain boundaries.
- **Upper-Right:** MSLP isobars (black contours, $\text{hPa}$) overlaid with 10m wind speed (color-fill) and wind barbs (knots).
- **Lower-Left:** 2m Dewpoint ($^\circ\text{C}$) displaying moisture gradients.
- **Lower-Right:** 6-Hour Precipitation accumulation ($\text{mm}$) with elevation terrain contours (brown lines).

---

## 💻 Hardware & Resource Management

The system is configured via `HybridComputeManager` in `model_architecture.py`:
- **NVIDIA GPU Detection:** If a CUDA-capable GPU is present, operations are automatically shifted to device `cuda:0` with FP16 Automatic Mixed Precision (`torch.cuda.amp`) enabled.
- **Apple Silicon Support:** Detects Metal Performance Shaders (`mps`) when running on macOS ARM devices.
- **Multi-Core CPU Tuning:** When running on CPU, the thread pool is pinned to $(N_{\text{cores}} - 1)$ to maximize throughput without freezing system processes.
- **Dynamic OOM Protection:** If an unusually fine resolution (e.g., $0.01^\circ$) exhausts GPU VRAM, the `safe_forward()` wrapper catches the CUDA Out-Of-Memory exception, clears GPU cache, flushes tensors to system RAM, and continues rollout on CPU without terminating the forecast.

---

## ⚠️ Troubleshooting & Operational Notes

1. **`eccodes` not found / missing shared library:**
   Ensure `eccodes` and `cfgrib` were installed from the `conda-forge` channel. If running on air-gapped systems without internet, pre-compile `eccodes` or supply pre-built wheels matching your OS distribution.

2. **Longitude Coordinate Discrepancies ($0^\circ \text{--} 360^\circ$ vs $-180^\circ \text{--} 180^\circ$):**
   Global grids frequently store longitudes from $0.0$ to $359.75$. The function `load_and_regrid_grib2` automatically detects if your bounding box requires western negative longitudes and normalizes the source coordinate system using `(lon + 180) % 360 - 180`.

3. **Sub-Grid Instability Across Complex Topography:**
   When using very high spatial resolutions ($< 0.03^\circ$) over steep mountain ranges, check the orographic lapse-rate adjustment in `forecast_engine.py` to ensure local elevation gradients do not produce unphysical temperature drops.
