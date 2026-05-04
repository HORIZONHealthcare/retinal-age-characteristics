# Data

Training, validation and test data for the retinal age model are passed to `code/train.py` via CSV manifests. Each row is one image; each CSV must have exactly three columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | string | Unique identifier of the eye / record (used for grouping in downstream cluster bootstrap). |
| `image_path` | string | Absolute path to a colour fundus photograph (CFP) on the host filesystem. |
| `biomarker_value` | float | Chronological age of the patient in years at the time of imaging. |

See [`example.csv`](example.csv) for the file format. The example contains synthetic rows only; replace `image_path` with paths to your own CFP files.

## Suggested directory layout

You typically need three CSVs (train / val / test) per dataset, e.g.

```
my_data/
├── train.csv
├── val.csv
└── test.csv
```

and pass `--splits_dir my_data` plus (optional) `--train_csv_name`, `--val_csv_name`, `--test_csv_name` to `train.py`.

## Image preparation

CFPs should be:
- pre-processed to remove black borders (square-padded);
- saved at any resolution ≥ the model input (typically 224 × 224 or larger);
- readable by `PIL.Image.open(...).convert("RGB")`.

The dataloader in `code/util/datasets.py` resizes every image to `backbone_img_size` (224 by default) and applies ImageNet-normalisation; no external preprocessing is required at training / inference time beyond the PIL-readable RGB constraint.

The cohort filtering, healthy/unhealthy labelling and split-assembly used in the manuscript are not included here: they involve site-specific clinical linkage that cannot be released. Re-using the manifest format above with your own splits is sufficient to reproduce all retinal-age-model quantities (MAE, RMSE, mean RAG, hexbin plots).
