from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_SEEDS,
    append_csv,
    build_pretrain_row,
    configure_tabred,
    dump_run_config,
    evaluate_saved_predictions,
    export_config,
    format_terminal_metrics,
    import_train_function,
    load_exported_config,
    parse_int_csv_arg,
    pretrain_row_header,
    resolve_datasets,
    resolve_models,
    update_group_stats,
)
from .paths import resolve_artifacts_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain TabReD classification baselines from HPO best configs.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--tabred_root", default=None)
    parser.add_argument("--artifacts_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_epochs_override", type=int, default=None)
    parser.add_argument("--patience_override", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> int:
    configure_tabred(args.tabred_root)
    artifacts_root = resolve_artifacts_root(args.artifacts_root)
    pretrain_root = artifacts_root / "pretrain"
    models = resolve_models(args.models)
    datasets = resolve_datasets(args.datasets)
    seeds = parse_int_csv_arg(args.seeds)

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    failures: list[tuple[str, str, int, str]] = []
    overall_csv = pretrain_root / "pretrain_seed_results.csv"
    overall_stats_csv = pretrain_root / "pretrain_seed_results_stats.csv"
    tasks = [(dataset_name, model_name, seed) for dataset_name in datasets for model_name in models for seed in seeds]

    for task_idx, (dataset_name, model_name, seed) in enumerate(tasks):
        if task_idx % args.num_shards != args.shard_index:
            continue

        print(
            f"[pretrain] shard={args.shard_index}/{args.num_shards} task={task_idx} "
            f"dataset={dataset_name} model={model_name} seed={seed}"
        )
        if args.dry_run:
            continue

        try:
            export_config(model_name, dataset_name, tabred_root=args.tabred_root, artifacts_root=artifacts_root)
            exported = load_exported_config(model_name, dataset_name, artifacts_root=artifacts_root)
            train_fn = import_train_function(exported.train_function)
            run_config = copy.deepcopy(exported.config)
            run_config["seed"] = int(seed)
            if args.n_epochs_override is not None:
                run_config["n_epochs"] = int(args.n_epochs_override)
            if args.patience_override is not None:
                run_config["patience"] = int(args.patience_override)
            if args.batch_size_override is not None:
                run_config["batch_size"] = int(args.batch_size_override)

            output_dir = pretrain_root / dataset_name / model_name / f"seed{seed}"
            dump_run_config(output_dir, run_config)
            train_fn(run_config, output_dir, force=args.force)

            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            metrics = evaluate_saved_predictions(output_dir, run_config, report, args.split)
            row = build_pretrain_row(
                dataset_name=dataset_name,
                model_name=model_name,
                seed=seed,
                split=args.split,
                output_dir=output_dir,
                checkpoint_path=output_dir / "checkpoint.pt",
                train_function=exported.train_function,
                metrics=metrics,
            )
            print(
                f"[pretrain-result] dataset={dataset_name} model={model_name} "
                f"seed={seed} split={args.split} {format_terminal_metrics(metrics)}"
            )

            dataset_csv = pretrain_root / dataset_name / "pretrain_seed_results.csv"
            dataset_stats_csv = pretrain_root / dataset_name / "pretrain_seed_results_stats.csv"
            append_csv(dataset_csv, row, pretrain_row_header())
            update_group_stats(dataset_csv, dataset_stats_csv, "model")
            append_csv(overall_csv, row, pretrain_row_header())
            update_group_stats(overall_csv, overall_stats_csv, "model")
        except Exception as exc:  # noqa: BLE001
            failures.append((dataset_name, model_name, seed, str(exc)))
            print(f"[pretrain] FAILED dataset={dataset_name} model={model_name} seed={seed} err={exc}")
            if not args.continue_on_error:
                break

    print(f"[pretrain] outputs={pretrain_root}")
    print(f"[pretrain] summary={overall_csv}")
    if failures:
        print("[pretrain] failures")
        for dataset_name, model_name, seed, err in failures:
            print(f"  - dataset={dataset_name} model={model_name} seed={seed}: {err}")
        return 1
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
