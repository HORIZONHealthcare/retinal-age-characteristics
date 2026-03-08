# Retinal Age Characteristics

Retinal age prediction from fundus images using Vision Transformer (ViT) fine-tuned from DINOv2 pre-training. The model predicts chronological age from retinal photographs as a regression task, supporting both single-modality and multi-modality inputs.

## Project Structure

```
├── main_finetune.py          # Main entry: argument parsing, model init, training loop
├── engine_finetune.py        # Training and evaluation engine (per-epoch logic)
├── models_vit.py             # Model definitions (RETFound_dinov2, multi-modality variant)
├── util/
│   ├── datasets.py           # Dataset classes and data transforms
│   ├── lr_decay.py           # Layer-wise learning rate decay (BEiT-style)
│   ├── lr_sched.py           # Cosine learning rate scheduler with warmup
│   ├── pos_embed.py          # Positional embedding interpolation
│   └── misc.py               # Distributed training utilities, checkpointing, logging
├── scripts/
│   ├── base_qsub.sh          # SGE/PBS job submission template
│   ├── start_all.sh          # Batch job launcher across modalities and patient conditions
│   ├── test_only.sh          # Evaluation-only script (external dataset testing)
│   └── check_epoch.sh        # Training progress monitoring utility
```

## Model Architecture

- **Backbone**: `vit_large_patch14_dinov2.lvd142m` (ViT-Large, patch size 14, DINOv2 pre-trained) via [timm](https://github.com/huggingface/pytorch-image-models)
- **Pre-trained weights**: RETFound DINOv2 checkpoint (`RETFound_dinov2_meh.pth`), loaded from the teacher branch with key remapping
- **Prediction head**: Linear(1024, 32) → ReLU → Dropout(0.5) → Linear(32, 1)
- **Multi-modality variant** (`RETFound_dinov2_MM`): Separate ViT backbones per modality, features concatenated before the prediction head. Optional projector layers (Linear → LayerNorm → ReLU) for dimensionality reduction
- **Age normalization**: `(age - 50) / 50` for training targets, denormalized for evaluation

## Datasets

| Dataset | Description | Usage |
|---------|-------------|-------|
| **AlzEye** | Primary dataset with train/val/test splits | Training & evaluation |
| **UKB** | UK Biobank retinal images | External validation |
| **BRSET** | Brazilian Ophthalmological dataset | External validation |
| **MBRSET** | MBRSET retinal image dataset | External validation |

Patient conditions can be filtered: `mixed`, `healthy`, or `unhealthy`.

## Requirements

- Python 3.8+
- PyTorch 1.13+
- [timm](https://github.com/huggingface/pytorch-image-models)
- torchvision
- huggingface_hub
- pandas, numpy, matplotlib

## Usage

### Training

```bash
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py \
    --savemodel \
    --batch_size 32 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 \
    --input_size 224 \
    --modalities CFP \
    --train_mode finetune \
    --clip_grad 1.0 \
    --load_retfound \
    --mm_model \
    --patient_condition mixed \
    --task my_experiment
```

### Evaluation Only

```bash
python main_finetune.py \
    --eval \
    --input_size 224 \
    --modalities CFP \
    --mm_model \
    --output_dir ./outputs/ \
    --task my_experiment \
    --resume ./outputs/my_experiment/checkpoint.pth \
    --dataset_name ukb
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 128 | Batch size per GPU |
| `--epochs` | 50 | Total training epochs |
| `--blr` | 5e-3 | Base learning rate (scaled by effective batch size / 256) |
| `--layer_decay` | 0.65 | Layer-wise LR decay factor |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--input_size` | 224 | Input image resolution |
| `--warmup_epochs` | 10 | Learning rate warmup epochs |
| `--modalities` | CFP | Image modality (e.g., `CFP`, or `CFP+OCT` for multi-modal) |
| `--train_mode` | finetune | Training mode: `finetune` or `linear_probe` |
| `--load_retfound` | False | Load RETFound pre-trained weights |
| `--mm_model` | False | Use multi-modality model |
| `--enable_projector` | False | Add projector layers in multi-modality model |
| `--patient_condition` | mixed | Patient subset: `mixed`, `healthy`, `unhealthy` |
| `--dataset_name` | alzeye | Dataset: `alzeye`, `ukb`, `brset`, `mbrset` |
| `--eval` | False | Evaluation-only mode |
| `--resume` | - | Path to checkpoint for resuming or evaluation |

## Training Details

- **Optimizer**: AdamW with layer-wise learning rate decay (following BEiT/ELECTRA)
- **LR schedule**: Cosine decay with linear warmup
- **Loss**: MSE (Mean Squared Error)
- **Mixed precision**: AMP (Automatic Mixed Precision) with GradScaler
- **Data augmentation** (training): AutoAugment (`rand-m9-mstd0.5-inc1`), Random Erasing (p=0.25)
- **Best model selection**: Based on validation MAE (lowest)

## Evaluation Metrics

- **ME** (Mean Error): Bias indicator (true - predicted)
- **MAE** (Mean Absolute Error): Primary metric for model selection
- **RMSE** (Root Mean Squared Error)

Results are saved to `outputs/<task>/metrics_{mode}.csv` and per-sample predictions to `outputs/<task>/Predictions_{mode}.csv`.

## Cluster Submission (SGE)

```bash
# Submit all experiments (modalities x patient conditions)
bash scripts/start_all.sh

# Monitor training progress
bash scripts/check_epoch.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.