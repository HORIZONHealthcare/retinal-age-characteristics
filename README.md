# Differential group RTM effects in retinal age models

Code and manuscript for the retinal age model used in the preprint [`docs/preprint.md`](docs/preprint.md).

The retinal age model takes a colour fundus photograph (CFP) and predicts chronological age via a vision-transformer (DINOv2 / DINOv3) backbone with a small regression head. From the prediction we compute the retinal age gap (RAG = predicted age − chronological age) and report mean RAG, MAE and RMSE.

## Repository layout

```
.
├── README.md                ← this file
├── LICENSE                  ← CC BY-NC 4.0
├── docs/
│   ├── preprint.md          ← Results section + figure captions
│   └── exported_figs/       ← rendered figures (PDF + PNG)
├── code/
│   ├── README.md            ← detailed model / training reference
│   ├── train.py             ← entry point: training and evaluation
│   ├── engine.py            ← per-epoch train / evaluate loops, metric I/O
│   ├── models.py            ← BiomarkerModel (ViT backbone + regression head)
│   ├── config/default.yaml  ← default hyperparameters
│   ├── util/                ← dataloader, transforms, lr schedule, distributed utils
│   └── scripts/run.sh       ← single-GPU torchrun launcher
└── data/
    ├── README.md            ← CSV manifest format documentation
    └── example.csv          ← three-column synthetic example
```

The downstream analyses described in the preprint (logistic regressions, RTM-slope analysis, calibration, external evaluation) are not included in this repository, as they involve site-specific clinical metadata that cannot be released. The code here is sufficient to reproduce all retinal-age-model quantities (MAE, RMSE, mean RAG, hexbin) on any cohort where you can supply CFPs and chronological ages.

## Quick start

1. Install dependencies:
   ```bash
   pip install torch torchvision timm pandas numpy pyyaml tensorboard
   ```
2. Build train / val / test CSVs in the format described in [`data/README.md`](data/README.md). See [`data/example.csv`](data/example.csv).
3. Train:
   ```bash
   cd code
   bash scripts/run.sh \
       --exp_name my_run \
       --splits_dir ../data \
       --train_csv_name train.csv \
       --val_csv_name val.csv
   ```
4. Evaluate on a held-out test CSV:
   ```bash
   bash scripts/run.sh \
       --exp_name my_run \
       --resume logs/my_run/checkpoint.pth \
       --test_csv ../data/test.csv \
       --test_name internal \
       --eval
   ```

`logs/my_run/` will contain the checkpoint, per-epoch training stats, `metrics_*.csv` (epoch, ME, MAE, RMSE) and `predictions_*.csv` (subject_id, true_value, predicted_value). RAG = `predicted_value − true_value`; mean RAG = ME.

See [`code/README.md`](code/README.md) for the full training / evaluation reference, supported backbones, and config details.

## Citation

Paper preprint forthcoming. Please cite via the repository for now.

## License

Code and documentation are released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) — see [LICENSE](LICENSE).
