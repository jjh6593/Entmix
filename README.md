# EntMix TabReD Runner

이 저장소는 upstream TabReD를 `tabred/`에 두고, EntMix 쪽 코드만으로 TabReD classification
benchmark의 모델 사전학습, baseline 평가, NCTTA+EntMix TTA 평가를 실행하기 위한 재현 파이프라인입니다.

원본 TabReD 학습 코드는 수정하지 않습니다. 데이터는 `data/`, 모델 checkpoint와 결과는 `artifacts/`에
저장됩니다. 두 경로와 `tabred/`는 `.gitignore`에 포함되어 있습니다.

## Installation

TabReD 환경 생성 방식은 TabReD README의 environment 안내를 따릅니다.

```bash
git clone https://github.com/yandex-research/tabred.git tabred
cd tabred
git checkout 36352fc567f5fb396bfc55bdec04e3cdf923e941
cd ..

micromamba create -f environment.yml
micromamba activate entmix-tabred
pip install -e tabred
pip install -e .
```

또는 bootstrap script로 같은 작업을 수행할 수 있습니다.

```bash
python scripts/bootstrap_tabred.py
```

For Kaggle datasets, enroll the respective competitions and have a Kaggle account, as described in
the TabReD README.

## Dataset

기본 예시는 `ecom-offers`만 실행합니다. 다른 classification benchmark를 추가하려면 주석 처리된 값을
사용하세요.

```bash
DATASETS=ecom-offers
# DATASETS=ecom-offers,homecredit-default,homesite-insurance

python scripts/preprocess_datasets.py \
  --datasets "$DATASETS"
```

`tabred/data`는 `data/`로 연결되는 symlink입니다. TabReD 코드는 기존 `:data/...` 경로를 그대로 쓰고,
실제 데이터는 이 저장소의 `data/`에 저장됩니다.

## 1. Pretrain

기본 예시는 `MLP` 하나만 학습합니다. 다른 모델을 추가하려면 주석 처리된 값을 사용하세요.

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

출력:

- `artifacts/configs/<model>/<dataset>.toml`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/checkpoint.pt`
- `artifacts/pretrain/<dataset>/<model>/seed<seed>/report.json`
- `artifacts/pretrain/pretrain_seed_results.csv`

빠른 동작 확인만 하려면 epoch를 줄일 수 있습니다.

```bash
python scripts/run_pretrain.py \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --n_epochs_override 1 \
  --patience_override 1 \
  --force
```

## 2. Baseline + NCTTA+EntMix

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

출력:

- `artifacts/results/<dataset>/baseline/summary.csv`
- `artifacts/results/<dataset>/nctta_entmix/summary.csv`
- `artifacts/results/<dataset>/method_summary.csv`
- `artifacts/results/all_summary_compact_runs.csv`

기본 NCTTA+EntMix hyperparameter는 `2026-04-12 latest_hpo_best_bacc_full` 선택값을
dataset/model별로 내장했습니다. 다른 선택 파일을 쓰려면:

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

단계별 실행:

```bash
python scripts/run_pipeline.py --stages setup
python scripts/run_pipeline.py --stages preprocess --datasets ecom-offers
python scripts/run_pipeline.py --stages pretrain --datasets ecom-offers --models mlp --seeds 0 --force_pretrain
python scripts/run_pipeline.py --stages tta --datasets ecom-offers --models mlp --seeds 0
```

## TabReD 사용

이 구조에서는 `tabred/`를 사용해도 됩니다. 다만 GitHub에 올릴 때 선택지는 둘입니다.

1. `tabred/`를 `.gitignore`에 둔다.
   - 현재 기본 방식입니다.
   - README 또는 bootstrap script로 upstream TabReD를 다시 받게 합니다.

2. `tabred/`를 포함해서 배포한다.
   - TabReD는 Apache-2.0이므로 사용과 재배포가 가능합니다.
   - 원본 `LICENSE`와 attribution을 유지해야 합니다.
   - 수정한 TabReD 파일이 있다면 수정 사실을 표시해야 합니다.
   - 데이터셋, checkpoint, prediction 같은 산출물은 별도 라이선스/용량 문제가 있으므로 포함하지 않습니다.

## GitHub

```bash
git add README.md LICENSE THIRD_PARTY_NOTICES.md .gitignore pyproject.toml environment.yml entmix scripts
git commit -m "Add EntMix TabReD reproduction pipeline"
git remote add origin git@github.com:<your-id>/<your-repo>.git
git push -u origin main
```

Git에 올리지 않는 경로:

- `tabred/`
- `data/`
- `artifacts/`
- `*.pt`, `*.npz`, `*.npy`, `*.parquet`, `*.zip`, `*.csv`
