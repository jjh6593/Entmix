# EntMix

## Installation

```bash
git clone --recurse-submodules https://github.com/jjh6593/Entmix.git
cd Entmix

micromamba create -f environment.yml
micromamba activate entmix-tabred

pip install -e tabred
pip install -e .
python scripts/bootstrap_tabred.py
```

이미 clone한 뒤 `tabred/`가 비어 있다면:

```bash
git submodule update --init --recursive
python scripts/bootstrap_tabred.py
```

Kaggle datasets require the same setup described in the TabReD README.

## Dataset

```bash
DATASETS=ecom-offers
# DATASETS=ecom-offers,homecredit-default,homesite-insurance

python scripts/preprocess_datasets.py \
  --datasets "$DATASETS"
```

## 1. Pretrain

```bash
DATASETS=ecom-offers
# DATASETS=ecom-offers,homecredit-default,homesite-insurance

MODELS=mlp
# MODELS=mlp,snn,dcn2,resnet,ft_transformer

SEEDS=0
# SEEDS=0,1,2

python scripts/run_pretrain.py \
  --datasets "$DATASETS" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  --force
```

Outputs:

- `artifacts/configs/<model>/<dataset>.toml`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/checkpoint.pt`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/report.json`
- `artifacts/pretrain/pretrain_seed_results.csv`

Smoke test:

```bash
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
DATASETS=ecom-offers
# DATASETS=ecom-offers,homecredit-default,homesite-insurance

MODELS=mlp
# MODELS=mlp,snn,dcn2,resnet,ft_transformer

SEEDS=0
# SEEDS=0,1,2

python scripts/run_tta.py \
  --datasets "$DATASETS" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  --methods baseline,nctta_entmix
```

Outputs:

- `artifacts/results/<dataset>/baseline/summary.csv`
- `artifacts/results/<dataset>/nctta_entmix/summary.csv`
- `artifacts/results/<dataset>/method_summary.csv`
- `artifacts/results/all_summary_compact_runs.csv`

Use a selected config JSON:

```bash
python scripts/run_tta.py --selected_json path/to/selected_tta_configs.json
```

## Pipeline

```bash
DATASETS=ecom-offers
# DATASETS=ecom-offers,homecredit-default,homesite-insurance

MODELS=mlp
# MODELS=mlp,snn,dcn2,resnet,ft_transformer

SEEDS=0
# SEEDS=0,1,2

python scripts/run_pipeline.py \
  --stages setup,preprocess,pretrain,tta \
  --datasets "$DATASETS" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  --methods baseline,nctta_entmix \
  --force_pretrain
```

Step by step:

```bash
python scripts/run_pipeline.py --stages setup
python scripts/run_pipeline.py --stages preprocess --datasets ecom-offers
python scripts/run_pipeline.py --stages pretrain --datasets ecom-offers --models mlp --seeds 0 --force_pretrain
python scripts/run_pipeline.py --stages tta --datasets ecom-offers --models mlp --seeds 0
```

## Notes

- `tabred/` is a Git submodule.
- `data/` and `artifacts/` are ignored by Git.
- Model files, predictions, datasets, and result CSV files are not committed.
- See `THIRD_PARTY_NOTICES.md` for TabReD license notes.
