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
- Added `MSSA_PLUS_HYBRID`, which keeps the 8-class branch classifier and adds a
  3-axis structured head for axis4/axis5/axis6 compound-fault decomposition.

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

## MSSA_PLUS Trial

`models/MSSA_PLUS.py` adds confidence-weighted ensemble pseudo labels for LMMD,
source reliability weighting, target classifier-consistency regularization, MCC
target regularization, and optional source label smoothing. The command below used
the same aligned ROBOT split as the MSSA run:

```shell
python train.py \
  --model_name MSSA_PLUS \
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
  --save_dir ./ckpt_robot_mssaplus_aligned_1024
```

Best MSSA_PLUS result from `_best.pth`:

| Metric | Value | Delta vs MSSA |
| --- | ---: | ---: |
| best epoch | 7 | -4 epochs |
| accuracy | 33.08% | -5.99 pp |
| macro F1 | 30.24% | -4.85 pp |
| weighted F1 | 31.01% | -4.78 pp |

This default MSSA_PLUS setting did not improve over the aligned MSSA baseline.
The training source accuracy rose to 93.87% by epoch 30, while target accuracy
settled around 30%, so the added pseudo-label, consistency, and MCC terms appear
too aggressive for this split without tuning.


## MSSA_PLUS_HYBRID Trial

`models/MSSA_PLUS_HYBRID.py` extends `MSSA_PLUS` with a 3-output axis head. The
head maps the ROBOT 8-class label powerset into independent axis4/axis5/axis6
fault probabilities, adds source axis BCE/NLL losses, and blends the 8-class
softmax head with the class probabilities implied by the axis head at inference.

```shell
python train.py \
  --model_name MSSA_PLUS_HYBRID \
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
  --save_dir ./ckpt_robot_mssaplus_hybrid_aligned_1024
```

Best MSSA_PLUS_HYBRID result from `_best.pth`:

| Metric | Value | Delta vs MSSA | Delta vs MSSA_PLUS |
| --- | ---: | ---: | ---: |
| best epoch | 21 | +10 epochs | +14 epochs |
| accuracy | 28.24% | -10.83 pp | -4.84 pp |
| macro F1 | 27.69% | -7.40 pp | -2.55 pp |
| weighted F1 | 28.49% | -7.30 pp | -2.52 pp |

The HYBRID version did not improve the aligned target result. Source-domain
training accuracy reached 95.68% by epoch 30, but target validation peaked at
28.24%, which suggests the axis-decoupled auxiliary head currently adds source
fit without improving transfer on this ROBOT_0/ROBOT_1 -> ROBOT_2 split.

Checkpoints and raw logs are not committed to Git. Re-run the command above to
recreate them locally.
