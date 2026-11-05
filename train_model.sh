python train_unet.py \
  --use-cds \
  --start-year 2022 --end-year 2023 \
  --lat-min 25.0 --lat-max 35.0 \
  --lon-min -90.0 --lon-max -80.0 \
  --res 0.10 \
  --epochs 20 \
  --save-weights ./checkpoints/theater_trained.pt
