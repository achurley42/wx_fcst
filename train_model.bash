python train_unet.py \
  --grib-dir ./historical_grib2 \
  --lat-min 24.0 --lat-max 32.0 \
  --lon-min -88.0 --lon-max -78.0 \
  --res 0.05 \
  --epochs 20 \
  --save-weights ./checkpoints/gulf_coast_unet.pt
