# Code — Biomarker Regression

Retinal biomarker regression from colour fundus photographs (CFP) using DINOv2 / DINOv3 vision-transformer backbones.

## Directory Structure

```
code/
├── train.py              # Entry point: training, evaluation, checkpoint resume
├── models.py             # BiomarkerModel (backbone + prediction head)
├── engine.py             # train_one_epoch() / evaluate()
├── config/
│   └── default.yaml      # Default hyperparameters (YAML, nested keys)
├── util/
│   ├── datasets.py       # BiomarkerDataset, transforms, denormalization
│   ├── misc.py           # Distributed training utils, MetricLogger, checkpoint I/O
│   ├── lr_decay.py       # Layer-wise learning rate decay (BEiT-style)
│   └── lr_sched.py       # Cosine annealing with linear warmup
└── scripts/
    └── run.sh            # Single-GPU torchrun launcher
```

## Model Architecture

```
Input Image
    │
    ▼
┌──────────────────┐
│  ViT Backbone    │  DINOv2 (patch14) or DINOv3 (patch16), pretrained from timm
│  (frozen / fine- │
│   tuned)         │
└────────┬─────────┘
         │  patch tokens (exclude CLS & register tokens)
         ▼
    Mean Pooling → fc_norm
         │
         ▼
┌──────────────────┐
│  PredictionHead  │  Linear(embed_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1)
└────────┬─────────┘
         │
         ▼
   Scalar prediction (normalised biomarker value)
```

### Supported Backbones

| Family | Sizes | Patch | timm ID pattern |
|--------|-------|-------|-----------------|
| DINOv2 | small / base / large / giant | 14 | `vit_{size}_patch14_dinov2.lvd142m` |
| DINOv3 | small / base / large / huge  | 16 | `vit_{size}_patch16_dinov3.lvd1689m` |

### Training Modes

| Mode | Backbone | PredictionHead |
|------|----------|----------------|
| `linear_probe` | Frozen | Trainable |
| `finetune_last_k` | Last K blocks + fc_norm trainable, rest frozen | Trainable |
| `finetune_all` | All trainable (default) | Trainable |

PredictionHead is **always trainable** in all three modes.

## Configuration

Two-level config system: **YAML defaults** + **CLI overrides**.

Nested YAML keys are flattened with underscores: `backbone.version` → `--backbone_version`.

```yaml
# config/default.yaml (abridged)
backbone:
  version: "dinov2"
  model: "large"
  img_size: 224
head:
  hidden_dim: 32
  dropout: 0.5
train_mode: "finetune_all"
finetune_k: 4
batch_size: 32
epochs: 50
biomarker_mean: 50
biomarker_std: 50
```

Any YAML key can be overridden on the command line:

```bash
python train.py --config config/default.yaml --backbone_version dinov3 --epochs 100
```

## Data Format

The code expects CSV files with exactly three columns:

| Column | Description |
|--------|-------------|
| `subject_id` | Unique identifier (string) |
| `image_path` | Absolute path to CFP image |
| `biomarker_value` | Target value (numeric, e.g. age) |

See `data/README.md` for how to generate these CSVs from raw datasets.

## Usage

### Training

```bash
# Local single-GPU
cd code
bash scripts/run.sh \
    --exp_name dinov2_large_finetune \
    --splits_dir ../data/splits/alzeye \
    --train_csv_name train_healthy.csv \
    --val_csv_name val_healthy.csv \
    --train_mode finetune_all

# Explicit AlzEye split variants in the same directory
bash scripts/run.sh \
    --exp_name dinov2_large_finetune_mixed \
    --splits_dir ../data/splits/alzeye \
    --train_csv_name train_mixed.csv \
    --val_csv_name val_mixed.csv \
    --train_mode finetune_all
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--splits_dir` | Directory containing split CSV files |
| `--train_csv_name` | Train CSV filename within `splits_dir` (default: `train_healthy.csv`) |
| `--val_csv_name` | Validation CSV filename within `splits_dir` (default: `val_healthy.csv`) |
| `--test_csv_name` | Test CSV filename within `splits_dir` (default: `test.csv`) |
| `--exp_name` | Experiment name (output saved to `logs/<exp_name>/`) |
| `--eval_output_suffix` | Optional suffix appended to evaluation output filenames to avoid overwriting existing predictions |
| `--train_mode` | `linear_probe` / `finetune_all` / `finetune_last_k` |
| `--finetune_k` | Number of last blocks to unfreeze (for `finetune_last_k`) |
| `--backbone_version` | `dinov2` or `dinov3` |
| `--backbone_model` | `small` / `base` / `large` / `giant` (dinov2) or `huge` (dinov3) |

### Resume Training

```bash
bash scripts/run.sh \
    --resume logs/dinov2_large_finetune/checkpoint.pth
```

### Evaluation (Internal Test Set)

```bash
bash scripts/run.sh \
    --exp_name dinov2_large_finetune \
    --splits_dir ../data/splits/alzeye \
    --resume logs/dinov2_large_finetune/checkpoint.pth \
    --eval

# Evaluate with an explicit internal test filename in the same directory
bash scripts/run.sh \
    --exp_name dinov2_large_finetune \
    --splits_dir ../data/splits/alzeye \
    --test_csv_name test.csv \
    --resume logs/dinov2_large_finetune/checkpoint.pth \
    --eval
```

### Evaluation (External Datasets)

Use `--test_csv` to point to an external dataset. The dataset name is auto-derived from the CSV path (parent directory name), or set explicitly with `--test_name`.

```bash
# Evaluate on UKB — results saved as metrics_test_ukb_external_v1.csv / predictions_test_ukb_external_v1.csv
bash scripts/run.sh \
    --exp_name dinov2_large_finetune \
    --resume logs/dinov2_large_finetune/checkpoint.pth \
    --test_csv ../data/splits/ukb/test.csv \
    --test_name ukb \
    --eval_output_suffix external_v1 \
    --eval

# Evaluate on BRSET — results saved as metrics_test_brset_external_v1.csv / predictions_test_brset_external_v1.csv
bash scripts/run.sh \
    --exp_name dinov2_large_finetune \
    --resume logs/dinov2_large_finetune/checkpoint.pth \
    --test_csv ../data/splits/brset/test.csv \
    --test_name brset \
    --eval_output_suffix external_v1 \
    --eval

# Explicit dataset name
bash scripts/run.sh \
    --test_csv /some/custom/path.csv \
    --test_name my_dataset \
    --eval_output_suffix external_v1 \
    --eval --resume ...
```

## Output Structure

All outputs are saved under `logs/<exp_name>/`:

```
logs/<exp_name>/
├── checkpoint.pth           # Best model checkpoint
├── log.txt                  # Per-epoch training stats (JSON lines)
├── metrics_val.csv          # Validation metrics (epoch, ME, MAE, RMSE)
├── predictions_val.csv      # Validation predictions (subject_id, true_value, predicted_value)
├── metrics_test.csv         # Internal test metrics
├── predictions_test.csv     # Internal test predictions
├── metrics_test_ukb_external_v1.csv     # External: UKB metrics (example with suffix)
├── predictions_test_ukb_external_v1.csv # External: UKB predictions (example with suffix)
├── metrics_test_brset_external_v1.csv   # External: BRSET metrics (etc.)
└── events.out.tfevents.*    # TensorBoard logs
```

### Metrics

| Metric | Definition |
|--------|-----------|
| ME  (Mean Error) | `mean(predicted - true)` — positive = over-prediction |
| MAE (Mean Absolute Error) | `mean(|predicted - true|)` |
| RMSE (Root Mean Squared Error) | `sqrt(mean((predicted - true)²))` |

## AlzEye split variants

`data/splits/alzeye/` may contain multiple train/validation variants in the same directory:

- `train_healthy.csv`, `val_healthy.csv`
- `train_mixed.csv`, `val_mixed.csv`
- `train_unhealthy.csv`, `val_unhealthy.csv`
- `test.csv`
  - shared raw mixed-health internal test set

For the rebuilt AlzEye internal splits, you should pass `--train_csv_name` and `--val_csv_name` explicitly.
`test.csv` remains the shared internal test filename by default unless overridden by `--test_csv_name`.

Rich test CSVs are supported. The dataloader only requires:
- `subject_id`
- `image_path`
- `biomarker_value`

Any additional metadata columns in `test.csv` are ignored by inference/training but remain available for downstream analysis.

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.13
- [timm](https://github.com/huggingface/pytorch-image-models) (for ViT backbones)
- torchvision, pandas, numpy, PyYAML, tensorboard
