# Running with synthetic data (if you don't have a GRIB2 file handy)
python run_forecast.py \
  --lat-min 25.0 --lat-max 35.0 \
  --lon-min -90.0 --lon-max -80.0 \
  --res 0.1 \
  --outdir ./test_no_training
