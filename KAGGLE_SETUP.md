# Kaggle Setup

TabReD preprocessing scripts download datasets through the Kaggle API. Before
running dataset preprocessing, set up Kaggle authentication and accept the
required competition rules.

## 1. Create a Kaggle API Token

1. Log in to Kaggle.
2. Open Kaggle settings:
   https://www.kaggle.com/settings
3. In the API section, click `Create New Token`.
4. A file named `kaggle.json` will be downloaded.

## 2. Install the Token

Place `kaggle.json` in `~/.kaggle/`.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If the file was downloaded somewhere else, replace `~/Downloads/kaggle.json`
with the actual path.

## 3. Accept Competition Rules

For competition datasets, Kaggle requires accepting the rules on the competition
page before API downloads work.

For the default example dataset:

- Ecom Offers:
  https://www.kaggle.com/c/acquire-valued-shoppers-challenge

Optional additional classification datasets:

- Homecredit Default:
  https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
- Homesite Insurance:
  https://www.kaggle.com/competitions/homesite-quote-conversion

Open the relevant page while logged in and accept the competition rules/terms.

## 4. Check API Access

After installing `kaggle.json`, test the API:

```bash
kaggle competitions files acquire-valued-shoppers-challenge
```

If the command prints a file list, Kaggle access is ready.

## 5. Run Preprocessing

```bash
python scripts/preprocess_datasets.py \
  --datasets ecom-offers
```

Or through the full pipeline:

```bash
python scripts/run_pipeline.py \
  --stages setup,preprocess,pretrain,tta \
  --datasets ecom-offers \
  --models mlp \
  --seeds 0 \
  --methods baseline,nctta_entmix \
  --force_pretrain
```

## Troubleshooting

If Kaggle says the dataset cannot be downloaded:

- Check that `~/.kaggle/kaggle.json` exists.
- Check file permissions with `ls -l ~/.kaggle/kaggle.json`.
- Run `chmod 600 ~/.kaggle/kaggle.json`.
- Make sure the relevant competition rules were accepted in the browser.
- Confirm the environment has the Kaggle package installed:

```bash
python -c "import kaggle; print(kaggle.__version__)"
```

The `kaggle.json` file contains credentials. Do not commit it to GitHub.
