python run_forecast.py \
  --grib ./current_analysis_00z.grib2 \
  --lat-min 24.0 --lat-max 32.0 \
  --lon-min -88.0 --lon-max -78.0 \
  --res 0.05 \
  --weights ./checkpoints/gulf_coast_unet.pt \
  --outdir ./gulf_coast_72hr_fcst
