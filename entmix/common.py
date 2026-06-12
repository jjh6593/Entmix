from __future__ import annotations

import csv
import importlib
import inspect
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import tomli_w
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .paths import resolve_artifacts_root
from .tabred import assert_tabred_ready

DEFAULT_MODELS = ("mlp", "snn", "dcn2", "resnet", "ft_transformer")
SUPPORTED_MODELS = DEFAULT_MODELS
DEFAULT_DATASETS = ("ecom-offers", "homecredit-default", "homesite-insurance")
DEFAULT_SEEDS = (0, 1, 2)
METRIC_ORDER = ("acc", "bacc", "f1", "macrof1", "aucroc", "precision", "recall")
TERMINAL_METRICS = ("acc", "bacc", "f1", "macrof1")
EPS = 1e-8

MODEL_ALIASES = {
    "mlp": "mlp",
    "snn": "snn",
    "dcn2": "dcn2",
    "dcnv2": "dcn2",
    "resnet": "resnet",
    "ft-transformer": "ft_transformer",
    "ft_transformer": "ft_transformer",
    "fttransformer": "ft_transformer",
}

DATASET_ALIASES = {
    "ecom-offers": "ecom-offers",
    "ecom_offers": "ecom-offers",
    "ecomoffers": "ecom-offers",
    "homecredit": "homecredit-default",
    "homecredit-default": "homecredit-default",
    "homecredit_default": "homecredit-default",
    "homecreditdefault": "homecredit-default",
    "homesite": "homesite-insurance",
    "homesite-default": "homesite-insurance",
    "homesite-insurance": "homesite-insurance",
    "homesite_insurance": "homesite-insurance",
    "homesiteinsurance": "homesite-insurance",
}

DISPLAY_NAMES = {
    "mlp": "MLP",
    "snn": "SNN",
    "dcn2": "DCNv2",
    "resnet": "ResNet",
    "ft_transformer": "FT-Transformer",
    "ecom-offers": "Ecom Offers",
    "homecredit-default": "Homecredit Default",
    "homesite-insurance": "Homesite Insurance",
}

SUPPORTED_TRAIN_FUNCTIONS = {"bin.nn_baselines.main"}
METHOD_ORDER = {"baseline": 0, "unadapt": 0, "nctta_entmix": 1, "nctta_dple": 2}


@dataclass(frozen=True)
class ExportedConfig:
    model_name: str
    dataset_name: str
    config_path: Path
    meta_path: Path
    train_function: str
    source_report_path: Path
    source_config_path: Optional[Path]
    source_kind: str
    config: dict[str, Any]


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_int_csv_arg(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def canonicalize_model(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"unsupported model {name!r}. supported={sorted(SUPPORTED_MODELS)}")
    return MODEL_ALIASES[key]


def canonicalize_dataset(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("_", "-")
    if key not in DATASET_ALIASES:
        raise ValueError(f"unsupported dataset {name!r}. supported={sorted(DEFAULT_DATASETS)}")
    return DATASET_ALIASES[key]


def resolve_models(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_MODELS)
    return [canonicalize_model(x) for x in parse_csv_arg(value)]


def resolve_datasets(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_DATASETS)
    return [canonicalize_dataset(x) for x in parse_csv_arg(value)]


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_toml(path: Path, config: Mapping[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(tomli_w.dumps(dict(config)), encoding="utf-8")


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any], header: Sequence[str]) -> None:
    ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def upsert_csv(path: Path, row: Mapping[str, Any], header: Sequence[str], key_fields: Sequence[str]) -> None:
    rows = read_rows(path)
    key = tuple(str(row.get(field, "")) for field in key_fields)
    replaced = False
    out: list[Mapping[str, Any]] = []
    for existing in rows:
        existing_key = tuple(str(existing.get(field, "")) for field in key_fields)
        if existing_key == key:
            out.append(dict(row))
            replaced = True
        else:
            out.append(existing)
    if not replaced:
        out.append(dict(row))
    write_rows(path, out, header)


def update_group_stats(summary_csv: Path, stats_csv: Path, group_key: str) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        return
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_group.setdefault(row[group_key], []).append(row)

    stats_rows: list[dict[str, Any]] = []
    metric_cols = [
        key
        for key in rows[0].keys()
        if key in METRIC_ORDER or key.endswith(tuple(f"_{m}" for m in METRIC_ORDER)) or "prior_" in key
    ]
    for group, group_rows in by_group.items():
        out: dict[str, Any] = {group_key: group, "n_runs": len(group_rows)}
        for col in metric_cols:
            vals: list[float] = []
            for row in group_rows:
                try:
                    vals.append(float(row[col]))
                except Exception:
                    continue
            if vals and not all(math.isnan(v) for v in vals):
                out[f"{col}_mean"] = float(np.nanmean(vals))
                out[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        stats_rows.append(out)
    if stats_rows:
        header = sorted({key for row in stats_rows for key in row.keys()})
        write_rows(stats_csv, stats_rows, header)


def select_known_kwargs(func: Callable[..., Any], values: Mapping[str, Any]) -> dict[str, Any]:
    params = set(inspect.signature(func).parameters)
    return {key: value for key, value in values.items() if key in params}


def source_paths_for_best_config(tabred_root: Path, model_name: str, dataset_name: str) -> tuple[Path, Path | None, str, dict[str, Any], str]:
    exp_dir = tabred_root / "exp" / model_name / dataset_name
    tuning_report = exp_dir / "tuning" / "report.json"
    if tuning_report.exists():
        payload = json.loads(tuning_report.read_text(encoding="utf-8"))
        best = payload["best"]
        config = best["config"]
        train_function = best.get("function") or payload.get("config", {}).get("function")
        if not train_function:
            raise ValueError(f"train function missing in {tuning_report}")
        return tuning_report, exp_dir / "tuning.toml", "tuning", config, str(train_function)

    eval_report = exp_dir / "evaluation" / "0" / "report.json"
    eval_config = exp_dir / "evaluation" / "0.toml"
    if eval_report.exists():
        payload = json.loads(eval_report.read_text(encoding="utf-8"))
        config = payload.get("config")
        if not isinstance(config, dict):
            if not eval_config.exists():
                raise FileNotFoundError(f"fallback config missing for {eval_report}")
            config = load_toml(eval_config)
        train_function = payload.get("function")
        if not train_function:
            raise ValueError(f"train function missing in {eval_report}")
        return eval_report, eval_config if eval_config.exists() else None, "evaluation0", config, str(train_function)

    raise FileNotFoundError(f"no tuning/evaluation report found for {model_name}/{dataset_name}")


def export_config(
    model_name: str,
    dataset_name: str,
    tabred_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> ExportedConfig:
    root = assert_tabred_ready(tabred_root)
    artifacts = resolve_artifacts_root(artifacts_root)
    model_name = canonicalize_model(model_name)
    dataset_name = canonicalize_dataset(dataset_name)
    source_report, source_config, source_kind, config, train_function = source_paths_for_best_config(
        root, model_name, dataset_name
    )
    if train_function not in SUPPORTED_TRAIN_FUNCTIONS:
        raise ValueError(f"{model_name}/{dataset_name} uses unsupported train_function={train_function}")

    config_path = artifacts / "configs" / model_name / f"{dataset_name}.toml"
    meta_path = artifacts / "configs" / model_name / f"{dataset_name}.meta.json"
    save_toml(config_path, config)
    save_json(
        meta_path,
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "train_function": train_function,
            "source_kind": source_kind,
            "source_report_path": str(source_report),
            "source_config_path": "" if source_config is None else str(source_config),
            "tabred_root": str(root),
            "exported_at": now_str(),
        },
    )
    return ExportedConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        config_path=config_path,
        meta_path=meta_path,
        train_function=train_function,
        source_report_path=source_report,
        source_config_path=source_config,
        source_kind=source_kind,
        config=config,
    )


def load_exported_config(
    model_name: str,
    dataset_name: str,
    artifacts_root: str | Path | None = None,
) -> ExportedConfig:
    artifacts = resolve_artifacts_root(artifacts_root)
    model_name = canonicalize_model(model_name)
    dataset_name = canonicalize_dataset(dataset_name)
    config_path = artifacts / "configs" / model_name / f"{dataset_name}.toml"
    meta_path = artifacts / "configs" / model_name / f"{dataset_name}.meta.json"
    if not config_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing exported config/meta for {model_name}/{dataset_name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ExportedConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        config_path=config_path,
        meta_path=meta_path,
        train_function=str(meta["train_function"]),
        source_report_path=Path(meta["source_report_path"]),
        source_config_path=Path(meta["source_config_path"]) if meta.get("source_config_path") else None,
        source_kind=str(meta["source_kind"]),
        config=load_toml(config_path),
    )


def import_train_function(qualname: str):
    module_name, function_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def dump_run_config(output_dir: Path, config: Mapping[str, Any]) -> None:
    save_toml(output_dir.with_suffix(".toml"), config)


def map_binary_labels(y: np.ndarray) -> tuple[np.ndarray, dict[Any, int]]:
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"binary classification required, found labels={classes.tolist()}")
    mapping = {classes[0]: 0, classes[1]: 1}
    y_int = np.vectorize(mapping.get, otypes=[np.int64])(y)
    return y_int.astype(np.int64), mapping


def apply_label_mapping(y: np.ndarray, mapping: Mapping[Any, int]) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    return np.vectorize(mapping.get, otypes=[np.int64])(y).astype(np.int64)


def probs_to_pos_numpy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs.astype(float)
    if probs.ndim == 2 and probs.shape[1] == 1:
        return probs[:, 0].astype(float)
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1].astype(float)
    raise ValueError(f"unsupported probability shape={tuple(probs.shape)}")


def prediction_to_pos_probs(prediction: np.ndarray, prediction_type: str) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction_type == "labels":
        return prediction.reshape(-1).astype(float)
    if prediction_type == "probs":
        return probs_to_pos_numpy(prediction)
    if prediction_type == "logits":
        if prediction.ndim == 1:
            return 1.0 / (1.0 + np.exp(-prediction.reshape(-1)))
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            return 1.0 / (1.0 + np.exp(-prediction[:, 0]))
        if prediction.ndim == 2 and prediction.shape[1] == 2:
            shifted = prediction - prediction.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs[:, 1]
    raise ValueError(f"unsupported prediction_type={prediction_type} shape={tuple(prediction.shape)}")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macrof1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["aucroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["aucroc"] = float("nan")
    return metrics


def format_metric_value(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(value_f):
        return "nan"
    return f"{value_f:.5f}"


def format_terminal_metrics(metrics: Mapping[str, float], keys: Sequence[str] = TERMINAL_METRICS) -> str:
    return " ".join(f"{key}={format_metric_value(metrics.get(key, float('nan')))}" for key in keys)


def evaluate_saved_predictions(output_dir: Path, config: Mapping[str, Any], report: Mapping[str, Any], split: str) -> dict[str, float]:
    import lib

    dataset = lib.data.build_dataset(**select_known_kwargs(lib.data.build_dataset, dict(config["data"])))
    predictions = lib.load_predictions(output_dir)
    y_true, _ = map_binary_labels(dataset["y"][split])
    y_prob = prediction_to_pos_probs(predictions[split], str(report["prediction_type"]))
    return compute_binary_metrics(y_true, y_prob)


def pretrain_row_header() -> list[str]:
    return [
        "timestamp",
        "dataset",
        "dataset_display",
        "model",
        "model_display",
        "seed",
        "split",
        "output_dir",
        "checkpoint_path",
        "train_function",
        *METRIC_ORDER,
    ]


def build_pretrain_row(
    dataset_name: str,
    model_name: str,
    seed: int,
    split: str,
    output_dir: Path,
    checkpoint_path: Path,
    train_function: str,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": now_str(),
        "dataset": dataset_name,
        "dataset_display": display_name(dataset_name),
        "model": model_name,
        "model_display": display_name(model_name),
        "seed": int(seed),
        "split": split,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_function": train_function,
    }
    for key in METRIC_ORDER:
        row[key] = float(metrics.get(key, float("nan")))
    return row


def tta_row_header(include_adapted: bool = True) -> list[str]:
    header = [
        "timestamp",
        "dataset",
        "dataset_display",
        "model",
        "model_display",
        "seed",
        "split",
        "method",
        "checkpoint_path",
    ]
    if include_adapted:
        for key in METRIC_ORDER:
            header.append(f"baseline_{key}")
        for key in METRIC_ORDER:
            header.append(f"adapted_{key}")
        header.extend(["source_prior_0", "source_prior_1", "final_prior_0", "final_prior_1"])
    else:
        header.extend(METRIC_ORDER)
    return header


def tta_compact_run_header() -> list[str]:
    return ["dataset", "method", "model", "seed", *METRIC_ORDER]


def build_compact_row(dataset_name: str, method: str, model_name: str, seed: int, metrics: Mapping[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset_name, "method": method, "model": model_name, "seed": int(seed)}
    for key in METRIC_ORDER:
        row[key] = float(metrics.get(key, float("nan")))
    return row


def write_dataset_method_summary(results_root: Path, dataset_name: str) -> None:
    dataset_root = results_root / dataset_name
    summary_rows: list[dict[str, Any]] = []
    for path in sorted(dataset_root.glob("*/summary_compact_runs.csv")):
        rows = read_rows(path)
        if not rows:
            continue
        method = rows[0].get("method", path.parent.name)
        out: dict[str, Any] = {"dataset": dataset_name, "method": method}
        for metric_name in METRIC_ORDER:
            values: list[float] = []
            for row in rows:
                try:
                    values.append(float(row[metric_name]))
                except Exception:
                    continue
            out[metric_name] = "nan" if not values else f"{float(np.nanmean(values)):.5f}"
        summary_rows.append(out)
    summary_rows.sort(key=lambda row: (METHOD_ORDER.get(str(row["method"]), 999), str(row["method"])))
    if summary_rows:
        write_rows(dataset_root / "method_summary.csv", summary_rows, ["dataset", "method", *METRIC_ORDER])


def configure_tabred(tabred_root: str | Path | None = None) -> Path:
    root = assert_tabred_ready(tabred_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import lib

    lib.configure_libraries()
    return root
