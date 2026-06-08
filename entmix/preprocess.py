from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import canonicalize_dataset
from .paths import DATA_ROOT
from .tabred import assert_tabred_ready, ensure_tabred_data_link

CLASSIFICATION_PREPROCESSING = {
    "ecom-offers": "ecom-offers.py",
    "homecredit-default": "homecredit.py",
    "homesite-insurance": "homesite.py",
}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def run_preprocessing(
    datasets: list[str],
    tabred_root: str | Path | None = None,
    force: bool = False,
) -> None:
    root = assert_tabred_ready(tabred_root)
    ensure_tabred_data_link(root, DATA_ROOT)
    for dataset in datasets:
        dataset = canonicalize_dataset(dataset)
        if dataset not in CLASSIFICATION_PREPROCESSING:
            raise ValueError(f"unsupported classification dataset: {dataset}")
        info_path = DATA_ROOT / dataset / "info.json"
        if info_path.exists() and not force:
            print(f"[preprocess] skip existing dataset={dataset} path={info_path.parent}")
            continue
        script = root / "preprocessing" / CLASSIFICATION_PREPROCESSING[dataset]
        print(f"[preprocess] dataset={dataset} script={script}")
        subprocess.run([sys.executable, str(script)], cwd=str(root), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and preprocess TabReD classification datasets.")
    parser.add_argument(
        "--datasets",
        default=",".join(CLASSIFICATION_PREPROCESSING),
        help="Comma-separated list: ecom-offers,homecredit-default,homesite-insurance",
    )
    parser.add_argument("--tabred_root", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_preprocessing(parse_csv(args.datasets), tabred_root=args.tabred_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
