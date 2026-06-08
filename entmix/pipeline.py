from __future__ import annotations

import argparse

from .common import DEFAULT_DATASETS, DEFAULT_MODELS, DEFAULT_SEEDS
from .preprocess import run_preprocessing
from .pretrain import run as run_pretrain
from .tabred import clone_or_checkout_tabred, ensure_tabred_data_link
from .tta import run as run_tta


def parse_stages(value: str) -> list[str]:
    allowed = {"setup", "preprocess", "pretrain", "tta"}
    stages = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    invalid = [x for x in stages if x not in allowed]
    if invalid:
        raise ValueError(f"unsupported stages={invalid}; allowed={sorted(allowed)}")
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Entmix TabReD experiment pipeline.")
    parser.add_argument("--stages", default="setup,preprocess,pretrain,tta")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--tabred_root", default=None)
    parser.add_argument("--artifacts_root", default=None)
    parser.add_argument("--selected_json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", default="baseline,nctta_entmix")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs_override", type=int, default=None)
    parser.add_argument("--patience_override", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--force_setup", action="store_true", default=False)
    parser.add_argument("--force_data", action="store_true", default=False)
    parser.add_argument("--force_pretrain", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stages = parse_stages(args.stages)

    if "setup" in stages:
        tabred_root = clone_or_checkout_tabred(tabred_root=args.tabred_root, force=args.force_setup)
        ensure_tabred_data_link(tabred_root)

    if "preprocess" in stages:
        run_preprocessing(
            [x.strip() for x in args.datasets.split(",") if x.strip()],
            tabred_root=args.tabred_root,
            force=args.force_data,
        )

    if "pretrain" in stages:
        pretrain_args = argparse.Namespace(
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
            tabred_root=args.tabred_root,
            artifacts_root=args.artifacts_root,
            split=args.split,
            n_epochs_override=args.n_epochs_override,
            patience_override=args.patience_override,
            batch_size_override=args.batch_size_override,
            force=args.force_pretrain,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        rc = run_pretrain(pretrain_args)
        if rc != 0:
            return rc

    if "tta" in stages:
        tta_args = argparse.Namespace(
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
            methods=args.methods,
            tabred_root=args.tabred_root,
            artifacts_root=args.artifacts_root,
            pretrain_root=None,
            results_root=None,
            selected_json=args.selected_json,
            batch_size=args.batch_size,
            split=args.split,
            device=args.device,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        rc = run_tta(tta_args)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
