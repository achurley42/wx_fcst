# Technical Guide: Monthly Weather Model Training

## Purpose and scope

`train_unet.py` trains and evaluates regional weather-model weights from six-hourly ERA5 and ERA5-Land history. The current workflow produces one calibration per **calendar month**, with regional metadata used when selecting weights for a forecast. A monthly weight file is not automatically considered successful: the script evaluates its 72-hour forecasts against ERA5 and records whether the calibration passed its configured thresholds.

The training entry point is `train_model.sh`. The current shell script requests CDS data for 2022–2023 over 25–35°N, 90–80°W at 0.1° resolution, with 20 training epochs per monthly model. These values can be changed in the shell script or by running `train_unet.py` with different arguments.

## End-to-end flow

```text
Parse settings and start a timestamped log
        |
        v
Check bounds-qualified monthly archives; download missing/empty months if --use-cds
        |
        v
For each archive: load its matching persistent regrid cache, or open/regrid GRIB and save cache
        |
        v
Sort frames by timestamp and form six-hour input -> target transitions
        |
        v
For each calendar month:
  train on earlier-year transitions for that month
  evaluate 72-hour validation rollouts in the latest year
  reduce learning rate and retrain for remaining rounds if thresholds fail
  select the best validation candidate and test it on a separate holdout
        |
        v
Save monthly weights and registry metadata as accepted or fallback
        |
        v
Forecast runner selects by initialization month and region overlap
```

## 1. Settings and logs

The script validates that epoch/round/run counts and RMSE thresholds are positive. It builds a bounding box from the command-line latitude and longitude bounds.

Standard output and standard error are mirrored to a log file:

- Default: `training_logs/train_<timestamp>.log`
- Override: `--log-file PATH`

The console and log contain the same training, ingestion, cache, validation, and error messages. The log file is opened in append mode.

## 2. Archive discovery and download

With `--use-cds`, the training code checks one archive per requested year and month. The filename includes the requested geographic bounds, year, and month, for example:

```text
era5_train_lat25_35_lon-90_-80_2022_01.grib2
```

A matching nonempty file is reused without parsing its GRIB contents. A missing or zero-byte file is downloaded from CDS. The monthly GRIB combines requested ERA5 surface fields, pressure-level fields, and ERA5-Land fields.

The check intentionally treats a nonempty bounds-qualified file as the requested archive; it does not prove that the contents are complete or readable. Existing generically named archives without bounds in the filename do not match and are not overwritten.

Without `--use-cds`, local GRIB files are selected from `--grib-dir`. In CDS mode, training subsequently loads only the expected bounds/year/month files, rather than unrelated GRIB files in the directory.

## 3. GRIB ingestion, regridding, and persistent cache

Each monthly archive is processed independently:

1. The cache key is computed from the archive's real path, byte size, modification time, requested bounds, target resolution, cache-format version, and channel names.
2. If a valid cache with that key exists, its frames and metadata are memory-mapped and reused. GRIB is not reopened and regridded for that archive.
3. Otherwise, cfgrib opens the archive; ingestion extracts the available valid times, selects the model variables and pressure levels, and interpolates fields onto the requested regular latitude/longitude grid.
4. Frames are normalized per channel using an archive-specific offset and scale, written as float16 to a disk-backed NumPy array, and accompanied by metadata containing timestamps, coordinates, offsets, and scales.
5. Once all archive records have been gathered, they are sorted by timestamp and presented to the dataset in chronological order.

The default cache directory is `~/.cache/wx_fcst/regrid`; `--cache-dir` overrides it. Cache files are kept outside the GRIB directory by default. A rerun reuses completed archive caches and rebuilds cache entries that are absent, mismatched, or structurally invalid. Changing source size or modification time, bounds, resolution, or channel schema results in a different cache key. The key does not hash every source-file byte.

Regridding can still take a long time on a cache miss. The cache only saves work after an archive has successfully been opened and regridded at least once.

## 4. Training sample construction

The dataset uses pairs of consecutive frames exactly six hours apart. Each pair is one training sample:

- **Input:** current six-hour frame plus derived static/time/location channels.
- **Target:** the next frame, represented as a correction to a simple baseline for most atmospheric variables.

Input channels total 34:

- 6 surface variables: 2 m temperature and dewpoint, 10 m u/v wind, mean sea-level pressure, and precipitation.
- 20 upper-air fields: geopotential, temperature, humidity, and u/v wind at 850, 700, 500, and 250 hPa.
- 4 additional channels: orography, land-sea mask, local-solar-time diurnal phase, and sine of latitude.
- 4 ERA5-Land fields: land skin temperature, first-layer soil temperature, first-layer soil moisture, and snow cover.

The weather and ERA5-Land input fields are standardized spatially for each sample/channel by subtracting that grid's mean and dividing by its standard deviation (with a small stabilizing constant). Orography is standardized similarly; the land-sea mask, diurnal phase, and latitude channel are supplied directly.

The target has six output channels. A simple baseline estimates advection using mean 850 hPa winds (or a fallback based on 10 m winds), shifts selected previous fields, applies a land-sea-dependent diurnal adjustment to temperature/dewpoint, and slightly damps 10 m wind over land. The model learns corrections to this baseline:

| Output | Training target |
|---|---|
| 2 m temperature | `(next ERA5 − baseline) / 0.5 K` |
| 2 m dewpoint | `(next ERA5 − baseline) / 0.3 K` |
| 10 m u wind | `(next ERA5 − baseline) / 0.4 m/s` |
| 10 m v wind | `(next ERA5 − baseline) / 0.4 m/s` |
| Mean sea-level pressure | `(next ERA5 − baseline) / 50 Pa` |
| Total precipitation | next ERA5 precipitation converted to mm (`× 1000`) |

Precipitation is a direct output target here; the code does not subtract a precipitation baseline in this training target.

## 5. Network and optimization

The network is `WeatherSpatioTemporalUNet`, a two-downsampling-level convolutional U-Net. It has encoder widths of 32 and 64 channels, a 128-channel bottleneck, skip connections into the decoder, and a 1×1 convolutional head producing the six correction fields. The input/output grid shape is preserved by padding to a multiple of four and cropping back to the original dimensions.

For each month, the script creates a new model. If `--resume-weights` points to an existing checkpoint, it initializes each month's model from that same checkpoint; otherwise the model starts from its initialized weights. `--epochs` is a **per-month** total epoch budget, distributed over up to `--max-calibration-rounds` rounds. If validation fails for a round, the learning rate is halved for the next round. The optimizer is AdamW, with weight decay `1e-4`; gradient norms are clipped at 1.0.

### What a training weight is

The model's weights are learned numeric parameters: convolution-kernel values and normalization parameters held in the layers of the U-Net. They are not per-variable multipliers or the ERA5 values themselves. Collectively, these parameters determine how the 34 input channels are transformed into six predicted correction grids. A saved checkpoint is a PyTorch `state_dict`, mapping parameter names to tensors; it contains the model parameters, not the training examples or optimizer state.

### How the weights change

For every training batch, the model runs a forward pass using its current parameters and produces six predicted correction fields. The loss compares these predictions with the target corrections and physical penalty. PyTorch then computes gradients of that loss with respect to the parameters (backpropagation). AdamW uses those gradients and its optimizer state to update the parameters. Gradient clipping limits the total gradient norm before the update.

This update cycle repeats for every batch and epoch. At the start of a new month, the script constructs a separate model, so monthly weights are trained independently. If a valid `--resume-weights` checkpoint is supplied, its model parameters initialize each month's model; otherwise each month starts from the architecture's normal initialization. This option loads weights only: it does not restore an optimizer, learning-rate schedule, epoch counter, or partially completed month.

When a monthly validation round fails, the model's updated parameters are retained for the next round, but a new AdamW optimizer is constructed at the reduced learning rate. Thus later rounds continue from the preceding round's model weights but do not continue the preceding optimizer's internal moment estimates.

The training loss is:

```text
latitude-weighted mean Smooth L1(output corrections, target corrections)
+ 2.5 × mean squared dewpoint-above-temperature violation
```

Latitude weights are proportional to cosine(latitude) and normalized to mean 1. The physical penalty is computed after converting the predicted temperature and dewpoint corrections back to physical values; it penalizes predicted dewpoint exceeding predicted temperature. The source variable named `mse_loss` is the averaged Smooth L1 term, not mean squared error.

Training loss is an optimization signal, not the calibration acceptance criterion.

## 6. Monthly training and holdout selection

The latest year present in the loaded dataset is held out from parameter training. For each calendar month:

- Training samples are six-hour transitions whose **target** timestamp is in that month and in a year earlier than the latest year.
- Candidate validation and test rollouts start in the latest year and in that month.
- A rollout candidate is eligible only if all reference timestamps from T+6h through T+72h exist in the dataset.
- Candidates are split in chronological order: the first 70% is the validation pool and the remaining portion is the test pool. Validation cases whose 72-hour window would overlap the first test start are removed.
- Up to `--validation-runs` validation cases and `--test-runs` test cases are spread across their respective pools.

Because validation rollouts start within a month, their final lead times can fall in the next calendar month. Training samples, by contrast, are assigned according to their target timestamp's month.

## 7. Verification, candidate choice, and saved status

Each validation rollout is a full 72-hour forecast. Verification compares predictions with ERA5 at T+6h, T+12h, …, T+72h for six outputs: temperature, dewpoint, both 10 m wind components, mean sea-level pressure, and precipitation. Pressure is compared in hPa and precipitation in mm.

Default RMSE limits are:

| Variable(s) | Limit |
|---|---:|
| 2 m temperature and dewpoint | 3 K |
| 10 m u/v wind | 3 m/s |
| Mean sea-level pressure | 5 hPa |
| Precipitation | 3 mm |

The limits can be changed with `--rmse-temperature-k`, `--rmse-wind-ms`, `--rmse-mslp-hpa`, and `--rmse-precip-mm`. A validation candidate passes only if every configured variable at every verification lead has RMSE at or below its limit.

After each round, the script snapshots the model parameters when that round is the best validation candidate so far. The score is the mean of `RMSE / configured limit` over all verification leads and variables. A candidate that passes validation is preferred over one that fails; among candidates with the same pass/fail status, the lower normalized validation score is preferred. When validation first passes, the script stops further training rounds, restores that round's saved parameter snapshot, and evaluates it on the independent test rollouts.

- **Accepted:** validation and independent test both pass every RMSE limit.
- **Fallback:** the selected candidate does not pass validation or the independent test. It is still saved and recorded so it can be used if no accepted calibration is available.
- **Incomplete:** there are no usable finite validation results, or the month lacks required training, validation, or test samples. The script reports these months as failures.

Test metrics are recorded for review but are not used to select among training rounds. If the chosen validation candidate fails the test, it is still saved as a fallback and marked accordingly; the test result does not trigger a search over other rounds. This avoids choosing a candidate by optimizing on the independent test set.

The `.pt` file is written from the selected model's `state_dict`. It does not contain the optimizer state or a resumable training-round record. The registry JSON stores the metadata and file path used to discover that checkpoint; it does not embed the tensors.

## 8. Monthly weights and forecast lookup

Weight paths are derived from `--save-weights`, with a three-letter month suffix; fallbacks also include `_fallback`. For example:

```text
checkpoints/regional_unet_best_jan.pt
checkpoints/regional_unet_best_feb_fallback.pt
```

The calibration registry stores the month number, status, normalized validation score, geographic bounds, weight path, history/validation years, rollout counts, thresholds, metrics, and input-channel count.

At forecast time, `run_forecast.py` uses the forecast initialization's UTC calendar month. It searches registry entries for that month and 34 input channels, requiring the calibration region to cover at least 50% of the requested forecast area. Accepted entries are preferred over fallback entries; among entries with the same status, greater region overlap is preferred, then lower validation score. If no compatible entry is found, the runner uses its normal baseline weights. Explicit `--weights` still overrides registry lookup.

Once selected, `forecast_engine.py` loads the checkpoint tensors into a newly constructed `WeatherSpatioTemporalUNet` before forecasting. The same parameter set is used at every autoregressive forecast hour; it is not retrained or updated during inference. The model's predicted corrections are combined with the forecast engine's baseline/advection calculations to produce the next weather state, which becomes input to the following hour.

Existing seasonal registry entries do not contain a month key and therefore are not selected by the monthly lookup. The registry file is upgraded to version 2 as monthly calibrations are saved; it is not automatically migrated into monthly calibrations.

## 9. Restart behavior and operational caveats

The persistent regrid cache is the completed preprocessing checkpoint. If training or calibration fails after preprocessing, rerunning with the same inputs can reuse each archive's completed cache and proceed to training without repeating interpolation. Calibration itself starts from freshly initialized weights on each rerun unless `--resume-weights` is explicitly supplied; the script does not automatically resume optimizer state or a partially completed calibration round.

Additional caveats:

- A nonempty archive is trusted by the download check; archive integrity and required field completeness are not checked there.
- If cfgrib cannot open an archive, the sequence loader currently warns and returns a synthetic frame without real timestamps. The training dataset filters out untimestamped frames. If sufficient other timestamped frames remain, training can continue with gaps; only six-hour adjacent pairs and complete validation windows are used.
- If no usable timestamped history remains in CDS mode, the dataset refuses to proceed with synthetic training data.
- Training and validation are regional, and monthly archive/regrid caches can consume substantial disk space.
- The default cache is under the current user's home directory. On WSL, it is under that WSL user's `~/.cache/wx_fcst/regrid`, not necessarily the Windows home directory.

## Main code locations

- `train_unet.py`: archive checks/download, persistent regrid cache, sample construction, model training, validation, and monthly checkpoint registration.
- `ingest_grib2.py`: GRIB opening, field selection, timestamp extraction, regridding, and missing-field behavior.
- `model_architecture.py`: U-Net architecture and compute-device selection.
- `calibration_registry.py`: monthly region matching and atomic registry updates.
- `run_forecast.py`: monthly registry lookup during inference.
