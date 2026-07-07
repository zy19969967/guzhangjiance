# ROBOT MSSA Experiment Notes

This project extends TL-Fault-Diagnosis-Library with a local ROBOT acceleration
dataset loader and MSSA settings for 18-channel robot-arm fault diagnosis.

## Code Changes

- Added `--in_channel` so MSSA can train on 18 acceleration channels instead of
  the original single-channel assumption.
- Added `--stride` for explicit sliding-window control.
- Added `--target_eval_mode` to support full-target evaluation or the original
  train/validation split behavior.
- Added a `ROBOT` loader for `.robot` pointer files and Excel workbooks.
- Skipped the generic post-window normalization for `ROBOT` because the loader
  already applies per-workbook per-channel min-max normalization to `[-1, 1]`.
- Saved a separate `_best.pth` checkpoint whenever validation accuracy improves.

## Full MSSA Run

The aligned full experiment used non-overlapping windows to match the MDSAN
windowing setup:

```shell
python train.py \
  --model_name MSSA \
  --source ROBOT_0,ROBOT_1 \
  --target ROBOT_2 \
  --data_dir datasets \
  --train_mode multi_source \
  --cuda_device 0 \
  --max_epoch 30 \
  --batch_size 64 \
  --signal_size 1024 \
  --stride 1024 \
  --in_channel 18 \
  --num_workers 0 \
  --opt adam \
  --lr 0.001 \
  --lr_scheduler fix \
  --target_eval_mode full \
  --save_dir ./ckpt_robot_aligned_1024
```

Dataset windows:

| Split | Windows |
| --- | ---: |
| source `ROBOT_0` (`old`) | 2,565 |
| source `ROBOT_1` (`new1`) | 2,568 |
| target adaptation `ROBOT_2` (`new2`) | 517 |
| target evaluation `ROBOT_2` (`new2`) | 517 |

Best MSSA result:

| Metric | Value |
| --- | ---: |
| best epoch | 11 |
| accuracy | 39.07% |
| macro F1 | 35.09% |
| weighted F1 | 35.79% |

For comparison, the local MDSAN multi-source experiment record was 30.17%
accuracy. The earlier MDSAN `old(10) -> new1(10)` baseline was 24.63% accuracy,
23.39% macro F1, and 23.65% weighted F1.

Checkpoints and raw logs are not committed to Git. Re-run the command above to
recreate them locally.
