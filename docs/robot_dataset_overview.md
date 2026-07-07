# ROBOT Acceleration Dataset Overview

This repository does not include the raw ROBOT dataset files. The raw Excel
workbooks are local experiment data and are excluded from Git by `.gitignore`.
This document records the dataset structure needed to reproduce the experiments.

## Signal Layout

Each Excel workbook contains a `Time` column plus 18 acceleration channels:

| Columns | Meaning |
| --- | --- |
| `AI1-01`, `AI1-02`, `AI1-03` | axis 1, X/Y/Z acceleration |
| `AI1-04`, `AI1-05`, `AI1-06` | axis 2, X/Y/Z acceleration |
| `AI1-07`, `AI1-08`, `AI1-09` | axis 3, X/Y/Z acceleration |
| `AI1-10`, `AI1-11`, `AI1-12` | axis 4, X/Y/Z acceleration |
| `AI1-13`, `AI1-14`, `AI1-15` | axis 5, X/Y/Z acceleration |
| `AI1-16`, `AI1-17`, `AI1-18` | axis 6, X/Y/Z acceleration |

The loader uses only the 18 acceleration channels and normalizes each workbook
per channel to `[-1, 1]`.

## Labels

| Label directory | Fault state |
| --- | --- |
| `00_normal` | normal |
| `01_axis4` | axis 4 fault |
| `02_axis5` | axis 5 fault |
| `03_axis6` | axis 6 fault |
| `04_axis45` | axis 4 + axis 5 fault |
| `05_axis46` | axis 4 + axis 6 fault |
| `06_axis56` | axis 5 + axis 6 fault |
| `07_axis456` | axis 4 + axis 5 + axis 6 fault |

## Domain and Speed Semantics

Filename tokens describe the domain and speed:

| Token | Meaning |
| --- | --- |
| `old` | trajectory/domain 0 |
| `new1` | trajectory/domain 1 |
| `new2` | trajectory/domain 2 |
| `new3` | trajectory/domain 3 |
| `(10)`, `(20)`, `(30)`, `(50)`, `(70)` | speed setting |

The local `new2` files do not include a speed token and are counted as unknown
speed.

## Local Dataset Counts

The local raw dataset used for the MSSA experiment contains 108 Excel workbooks
with a total size of about 813.7 MB.

| Class | Files |
| --- | ---: |
| `00_normal` | 16 |
| `01_axis4` | 11 |
| `02_axis5` | 16 |
| `03_axis6` | 16 |
| `04_axis45` | 11 |
| `05_axis46` | 11 |
| `06_axis56` | 16 |
| `07_axis456` | 11 |

| Trajectory/domain | Files |
| --- | ---: |
| `old` | 40 |
| `new1` | 40 |
| `new2` | 8 |
| `new3` | 20 |

| Speed | Files |
| --- | ---: |
| `10` | 20 |
| `20` | 20 |
| `30` | 20 |
| `50` | 20 |
| `70` | 20 |
| unknown | 8 |

## Rebuilding the Local ROBOT Index

Use the helper script to create a lightweight pointer dataset under
`datasets/ROBOT`. The generated `.robot` files are also excluded from Git
because they contain local absolute paths.

```shell
python tools/prepare_robot_dataset.py --raw_root E:/data/data --output_root datasets
```

For a smoke-test subset:

```shell
python tools/prepare_robot_dataset.py --raw_root E:/data/data --output_root datasets_smoke --max_files_per_class_condition 1
```

The resulting TL-Fault-Diagnosis-Library conditions are:

| Condition | Domain |
| --- | --- |
| `ROBOT_0` | `old` |
| `ROBOT_1` | `new1` |
| `ROBOT_2` | `new2` |
| `ROBOT_3` | `new3` |
