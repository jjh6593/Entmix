# EntMix

## Installation

```bash
git clone --recurse-submodules https://github.com/jjh6593/Entmix.git
cd Entmix

conda env create -f environment.yml
conda activate entmix-tabred

pip install -e .
python scripts/bootstrap_tabred.py
```

Kaggle datasets require API credentials. See `KAGGLE_SETUP.md`.

## Dataset

```bash
python scripts/preprocess_datasets.py \
  --datasets ecom-offers

# Add more datasets:
#   --datasets ecom-offers,homecredit-default,homesite-insurance
```

## 1. Pretrain

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pretrain.py \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --force

# Add more datasets/models/seeds:
#   --datasets ecom-offers,homecredit-default,homesite-insurance
#   --models mlp,snn,dcn2,resnet,ft_transformer
#   --seeds 0,1,2
```

Outputs:

- `artifacts/configs/<model>/<dataset>.toml`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/checkpoint.pt`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/report.json`
- `artifacts/pretrain/pretrain_seed_results.csv`

Smoke test:

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pretrain.py \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --n_epochs_override 1 \
  --patience_override 1 \
  --force
```

## 2. TTA

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/run_tta.py \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --methods baseline,nctta_entmix,nctta_dple

# Add more datasets/models/seeds:
#   --datasets ecom-offers,homecredit-default,homesite-insurance
#   --models mlp,snn,dcn2,resnet,ft_transformer
#   --seeds 0,1,2
#   --methods baseline,nctta_entmix,nctta_dple
```

Outputs:

- `artifacts/results/<dataset>/baseline/summary.csv`
- `artifacts/results/<dataset>/nctta_entmix/summary.csv`
- `artifacts/results/<dataset>/nctta_dple/summary.csv`
- `artifacts/results/<dataset>/method_summary.csv`
- `artifacts/results/all_summary_compact_runs.csv`

Use a selected config JSON:

```bash
python scripts/run_tta.py --selected_json path/to/selected_tta_configs.json
```

## Pipeline

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pipeline.py \
  --stages setup,preprocess,pretrain,tta \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --methods baseline,nctta_entmix,nctta_dple \
  --force_pretrain

# Add more datasets/models/seeds:
#   --datasets ecom-offers,homecredit-default,homesite-insurance
#   --models mlp,snn,dcn2,resnet,ft_transformer
#   --seeds 0,1,2
```

Step by step:

```bash
python scripts/run_pipeline.py --stages setup
python scripts/run_pipeline.py --stages preprocess --datasets ecom-offers
export CUDA_VISIBLE_DEVICES=0
python scripts/run_pipeline.py --stages pretrain --datasets ecom-offers --models mlp --seeds 0 --force_pretrain
python scripts/run_pipeline.py --stages tta --datasets ecom-offers --models mlp --seeds 0
```

## Notes

- `tabred/` is a Git submodule.
- When CUDA is available, set `CUDA_VISIBLE_DEVICES` explicitly before pretraining.
- `data/` and `artifacts/` are ignored by Git.
- Model files, predictions, datasets, and result CSV files are not committed.
- See `THIRD_PARTY_NOTICES.md` for TabReD notes.
