from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn.functional as F

from .common import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_SEEDS,
    EPS,
    METRIC_ORDER,
    apply_label_mapping,
    build_compact_row,
    compute_binary_metrics,
    configure_tabred,
    display_name,
    load_toml,
    now_str,
    parse_int_csv_arg,
    read_rows,
    resolve_datasets,
    resolve_models,
    select_known_kwargs,
    tta_compact_run_header,
    tta_row_header,
    update_group_stats,
    upsert_csv,
    write_dataset_method_summary,
    write_rows,
)
from .paths import resolve_artifacts_root


@dataclass
class SeedBundle:
    model_name: str
    dataset_name: str
    seed: int
    split: str
    output_dir: Path
    checkpoint_path: Path
    config: dict[str, Any]
    report: dict[str, Any]
    dataset: Any
    model: torch.nn.Module
    label_mapping: dict[Any, int]
    source_prior: torch.Tensor


@dataclass
class EntMixConfig:
    prior_alpha: float = 0.95
    prior_beta: float = 1.0
    entmix_min: float = 0.0
    entmix_max: float = 1.0
    target_prior_init: str = "uniform"


@dataclass
class NCTTAEntMixConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    steps_per_batch: int = 1
    alpha: float = 0.35
    top_k: int = 1
    distance_eps: float = 0.5
    tau_ent: float = 0.3
    entropy_threshold: float | None = 0.35
    nu: float = 1.0
    eta: float = 1.0
    adaptive_weight_mode: str = "intended_product"
    intended_product_nu_mode: str = "separate_multiplier"
    sample_selection_mode: str = "threshold"
    confidence_metric_mode: str = "entropy"
    entmix: EntMixConfig = field(default_factory=EntMixConfig)


COMMON_HPO_DEFAULTS: dict[str, Any] = {
    "batch_size_by_model": {"default": 512, "ft_transformer": 128},
    "entmix_max": 1.0,
    "entmix_min": 0.0,
    "entmix_prior_beta": 1.0,
    "nctta_adaptive_weight_mode": "intended_product",
    "nctta_alpha": 0.35,
    "nctta_confidence_metric_mode": "entropy",
    "nctta_distance_eps": 0.5,
    "nctta_eta": 1.0,
    "nctta_intended_product_nu_mode": "separate_multiplier",
    "nctta_nu": 1.0,
    "nctta_sample_selection_mode": "threshold",
    "nctta_steps_per_batch": 1,
    "nctta_tau_ent": 0.3,
    "nctta_top_k": 1,
    "nctta_weight_decay": 0.0,
}

HPO_BACC_20260412: dict[str, dict[str, dict[str, Any]]] = {
    "ecom-offers": {
        "mlp": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.611, "nctta_lr": 1e-4},
        "snn": {"entmix_prior_alpha": 0.8, "nctta_entropy_threshold": 0.45, "nctta_lr": 1e-4},
        "dcn2": {"entmix_prior_alpha": 0.9, "nctta_entropy_threshold": 0.53, "nctta_lr": 1e-3},
        "resnet": {"entmix_prior_alpha": 0.85, "nctta_entropy_threshold": 0.53, "nctta_lr": 1e-3},
        "ft_transformer": {"entmix_prior_alpha": 0.85, "nctta_entropy_threshold": 0.53, "nctta_lr": 1e-6},
    },
    "homecredit-default": {
        "mlp": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-6},
        "snn": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-5},
        "dcn2": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-5},
        "resnet": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-4},
        "ft_transformer": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-6},
    },
    "homesite-insurance": {
        "mlp": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.69, "nctta_lr": 1e-4},
        "snn": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-6},
        "dcn2": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-5},
        "resnet": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-3},
        "ft_transformer": {"entmix_prior_alpha": 0.95, "nctta_entropy_threshold": 0.35, "nctta_lr": 1e-6},
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_probs(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=EPS, posinf=EPS, neginf=EPS)
    x = x.clamp(min=EPS)
    return x / x.sum(dim=-1, keepdim=True).clamp(min=EPS)


def prob_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(min=EPS, max=1.0)
    return -(probs * probs.log()).sum(dim=1)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.shape[1] == 1:
        pos = torch.sigmoid(logits.squeeze(1))
        probs = torch.stack([1.0 - pos, pos], dim=1)
    elif logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)
    else:
        raise ValueError(f"binary logits expected, got shape={tuple(logits.shape)}")
    return normalize_probs(probs)


def map_binary_labels(y: np.ndarray) -> tuple[np.ndarray, dict[Any, int]]:
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"binary classification required, found labels={classes.tolist()}")
    mapping = {classes[0]: 0, classes[1]: 1}
    y_int = np.vectorize(mapping.get, otypes=[np.int64])(y)
    return y_int.astype(np.int64), mapping


def build_batches(
    dataset: Any,
    split: str,
    batch_size: int,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 0,
):
    y = dataset["y"][split]
    indices = np.arange(len(y))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch: dict[str, torch.Tensor] = {}
        if "x_num" in dataset:
            batch["x_num"] = torch.as_tensor(dataset["x_num"][split][idx], dtype=torch.float32, device=device)
        if "x_bin" in dataset:
            batch["x_bin"] = torch.as_tensor(dataset["x_bin"][split][idx], dtype=torch.float32, device=device)
        if "x_cat" in dataset:
            batch["x_cat"] = torch.as_tensor(dataset["x_cat"][split][idx], dtype=torch.long, device=device)
        batch_y = torch.as_tensor(dataset["y"][split][idx], device=device)
        yield batch, batch_y


def concat_batch_features(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if "x_num" in batch:
        pieces.append(batch["x_num"])
    if "x_bin" in batch:
        pieces.append(batch["x_bin"])
    if "x_cat" in batch:
        pieces.append(batch["x_cat"].float())
    if not pieces:
        raise ValueError("empty batch")
    return torch.cat(pieces, dim=1)


def _compute_bins_if_needed(dataset: Any, config: Mapping[str, Any]):
    bins_cfg = config.get("bins")
    if bins_cfg is None:
        return None
    bins_cfg = copy.deepcopy(bins_cfg)
    x_num_train = torch.as_tensor(dataset["x_num"]["train"], dtype=torch.float32)
    extra_kwargs: dict[str, Any] = {}
    if "tree_kwargs" in bins_cfg:
        y_train = torch.as_tensor(dataset["y"]["train"], dtype=torch.long)
        extra_kwargs = {"y": y_train, "regression": False, "verbose": True}
    return rtdl_num_embeddings.compute_bins(x_num_train, **bins_cfg, **extra_kwargs)


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(ckpt, tuple):
        state = ckpt[0]
    elif isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def build_model_from_config(config: Mapping[str, Any], device: torch.device):
    import lib
    from bin import nn_baselines

    data_cfg = select_known_kwargs(lib.data.build_dataset, copy.deepcopy(config["data"]))
    dataset = lib.data.build_dataset(**data_cfg)
    bins = _compute_bins_if_needed(dataset, config)
    model_cfg = copy.deepcopy(config["model"])
    model = nn_baselines.Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        bins=bins,
        **model_cfg,
    ).to(device)
    return dataset, model


def prepare_seed_bundle(
    pretrain_root: Path,
    dataset_name: str,
    model_name: str,
    seed: int,
    device: torch.device,
    split: str,
) -> SeedBundle:
    output_dir = pretrain_root / dataset_name / model_name / f"seed{seed}"
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    config_path = output_dir.with_suffix(".toml")
    for path in [output_dir, checkpoint_path, report_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(f"required pretrain artifact missing: {path}")

    config = load_toml(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset, model = build_model_from_config(config, device=device)
    model.load_state_dict(load_state_dict(checkpoint_path), strict=False)

    y_train_int, label_mapping = map_binary_labels(dataset["y"]["train"])
    counts = np.bincount(y_train_int, minlength=2).astype(np.float32)
    counts = np.clip(counts, EPS, None)
    source_prior = torch.as_tensor(counts / counts.sum(), dtype=torch.float32, device=device)
    return SeedBundle(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        split=split,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        report=report,
        dataset=dataset,
        model=model,
        label_mapping=label_mapping,
        source_prior=source_prior,
    )


def _prepare_flat_or_token_inputs(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    x_num = batch.get("x_num")
    x_bin = batch.get("x_bin")
    x_cat = batch.get("x_cat")
    if x_num is not None:
        parts.append(x_num if model.m_num is None else model.m_num(x_num))
    if x_bin is not None:
        parts.append(x_bin if model.m_bin is None else model.m_bin(x_bin))
    if x_cat is not None:
        if model.m_cat is None:
            raise ValueError("categorical batch received but model has no categorical encoder")
        if model.backbone.__class__.__name__ == "DCNv2":
            parts.append(model.m_cat(x_cat).flatten(-2))
        else:
            parts.append(model.m_cat(x_cat))
    if model.flat:
        return torch.column_stack([x.flatten(1, -1) for x in parts])
    return torch.cat([model.cls_embedding(parts[0].shape[:1])] + parts, dim=1)


def forward_logits_and_features(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    x = _prepare_flat_or_token_inputs(model, batch)
    backbone = model.backbone
    backbone_name = backbone.__class__.__name__

    if backbone_name == "MLP":
        hidden = x
        for block in backbone.blocks:
            hidden = block(hidden)
        logits = hidden if backbone.output is None else backbone.output(hidden)
        return logits, hidden

    if backbone_name in {"ResNet", "ResNetNoNorm"}:
        hidden = backbone.input_projection(x)
        for block in backbone.blocks:
            hidden = hidden + block(hidden)
        logits = hidden if backbone.output is None else backbone.output(hidden)
        return logits, hidden

    if backbone_name == "DCNv2":
        base = x
        hidden = x
        for cross in backbone.cross_layers:
            hidden = base * cross(hidden)
        hidden = backbone.deep_layers(hidden)
        logits = backbone.head(hidden)
        return logits, hidden

    if backbone_name == "SNN":
        hidden = x
        for block in backbone.blocks:
            hidden = block(hidden)
        logits = hidden if backbone.output is None else backbone.output(hidden)
        return logits, hidden

    if backbone_name == "FTTransformerBackbone":
        hidden = x
        n_blocks = len(backbone.blocks)
        for i_block, block in enumerate(backbone.blocks):
            x_identity = hidden
            x_norm = block["attention_normalization"](hidden) if "attention_normalization" in block else hidden
            x_attn = block["attention"](x_norm[:, :1] if i_block + 1 == n_blocks else x_norm, x_norm)
            x_attn = block["attention_residual_dropout"](x_attn)
            hidden = x_identity + x_attn

            x_identity = hidden
            x_ffn = block["ffn_normalization"](hidden)
            x_ffn = block["ffn"](x_ffn)
            x_ffn = block["ffn_residual_dropout"](x_ffn)
            hidden = x_identity + x_ffn
            hidden = block["output"](hidden)

        cls_hidden = hidden[:, 0]
        logits = cls_hidden if backbone.output is None else backbone.output(cls_hidden)
        return logits, cls_hidden

    raise ValueError(f"unsupported backbone={backbone_name}")


def get_classifier_weight(model: torch.nn.Module) -> torch.Tensor:
    backbone = model.backbone
    backbone_name = backbone.__class__.__name__
    if backbone_name == "MLP":
        weight = backbone.output.weight
    elif backbone_name in {"ResNet", "ResNetNoNorm"}:
        weight = backbone.output.linear.weight
    elif backbone_name == "DCNv2":
        weight = backbone.head.weight
    elif backbone_name == "SNN":
        weight = backbone.output.weight
    elif backbone_name == "FTTransformerBackbone":
        weight = backbone.output.linear.weight
    else:
        raise ValueError(f"unsupported backbone={backbone_name}")
    if weight.shape[0] == 1:
        return torch.cat([-weight, weight], dim=0)
    return weight


@torch.no_grad()
def run_baseline_eval(bundle: SeedBundle, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = copy.deepcopy(bundle.model).to(device)
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for batch, y in build_batches(bundle.dataset, bundle.split, batch_size, device=device, shuffle=False):
        logits, _ = forward_logits_and_features(model, batch)
        probs = logits_to_probs(logits)
        probs_all.append(probs[:, 1].detach().cpu().numpy().astype(np.float32))
        y_all.append(apply_label_mapping(y.detach().cpu().numpy(), bundle.label_mapping))
    return np.concatenate(probs_all), np.concatenate(y_all)


class EntMixPriorCorrector:
    def __init__(self, pi_s: torch.Tensor, cfg: EntMixConfig):
        self.pi_s = normalize_probs(pi_s.clone().detach())
        init_mode = str(cfg.target_prior_init).strip().lower()
        if init_mode == "uniform":
            self.pi_t = torch.ones_like(self.pi_s) / float(self.pi_s.numel())
        elif init_mode in {"source", "source_prior", "prior"}:
            self.pi_t = self.pi_s.clone()
        else:
            raise ValueError(f"unsupported EntMix target_prior_init={cfg.target_prior_init!r}")
        self.alpha = float(cfg.prior_alpha)
        self.beta = float(cfg.prior_beta)
        self.entmix_min = float(cfg.entmix_min)
        self.entmix_max = float(cfg.entmix_max)
        self.last_debug: dict[str, Any] = {}

    @torch.no_grad()
    def process_probs(self, probs: torch.Tensor, update_pi: bool = True) -> torch.Tensor:
        n_classes = probs.shape[1]
        uniform = torch.full((n_classes,), 1.0 / n_classes, device=probs.device)
        entropy = prob_entropy(probs)
        hbar = (entropy / np.log(n_classes)).clamp(0.0, 1.0)
        hbar = (hbar * (self.entmix_max - self.entmix_min) + self.entmix_min).clamp(0.0, 1.0)
        q = (1.0 - hbar).unsqueeze(1) * probs + hbar.unsqueeze(1) * uniform.unsqueeze(0)
        batch_pi = normalize_probs(q.mean(dim=0))
        if update_pi:
            self.pi_t = normalize_probs((1.0 - self.alpha) * batch_pi + self.alpha * self.pi_t)
        ratio = torch.clamp(self.pi_t / self.pi_s.clamp(min=EPS), min=EPS).pow(self.beta)
        return normalize_probs(probs * ratio.unsqueeze(0))


class NCTTAEntMixAdapter(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, source_prior: torch.Tensor, cfg: NCTTAEntMixConfig, device: torch.device):
        super().__init__()
        self.model = copy.deepcopy(base_model).to(device)
        self.device = device
        self.lr = float(cfg.lr)
        self.weight_decay = float(cfg.weight_decay)
        self.steps_per_batch = max(1, int(cfg.steps_per_batch))
        self.alpha = float(cfg.alpha)
        self.top_k = max(1, int(cfg.top_k))
        self.distance_eps = float(cfg.distance_eps)
        self.tau_ent = float(cfg.tau_ent)
        self.entropy_threshold = float(
            prob_entropy(torch.tensor([[0.7, 0.3]], device=device))[0].item()
            if cfg.entropy_threshold is None
            else cfg.entropy_threshold
        )
        self.nu = float(cfg.nu)
        self.eta = float(cfg.eta)
        self.adaptive_weight_mode = str(cfg.adaptive_weight_mode)
        self.intended_product_nu_mode = str(cfg.intended_product_nu_mode)
        self.sample_selection_mode = str(cfg.sample_selection_mode)
        self.confidence_metric_mode = str(cfg.confidence_metric_mode)
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.corrector = EntMixPriorCorrector(source_prior, cfg.entmix)

    def _get_features_and_probs(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = forward_logits_and_features(self.model, batch)
        return features, logits_to_probs(logits)

    def _fca(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_weights = get_classifier_weight(self.model)
        feat_norm = F.normalize(features, p=2, dim=1, eps=EPS)
        weight_norm = F.normalize(class_weights, p=2, dim=1, eps=EPS)
        similarity = feat_norm @ weight_norm.T
        distances = (feat_norm[:, None, :] - weight_norm[None, :, :]).pow(2).sum(dim=2)
        return distances, similarity

    def _multi_positive_nce(self, similarity: torch.Tensor, positive_idx: torch.Tensor) -> torch.Tensor:
        pos_mask = torch.zeros_like(similarity, dtype=torch.bool)
        pos_mask.scatter_(1, positive_idx, True)
        log_num = torch.logsumexp(similarity.masked_fill(~pos_mask, float("-inf")), dim=1)
        log_den = torch.logsumexp(similarity, dim=1)
        return -(log_num - log_den)

    def _neighbor_mask(self, batch: Mapping[str, torch.Tensor], probs: torch.Tensor) -> torch.Tensor:
        if self.sample_selection_mode == "none":
            return torch.ones(probs.shape[0], dtype=torch.float32, device=self.device)
        samples = concat_batch_features(batch).detach().cpu()
        pairwise = torch.cdist(samples, samples, p=2)
        distance_threshold = pairwise.mean()
        near_neighbor = (pairwise <= distance_threshold).float()
        pseudo = probs.detach().argmax(dim=1).float().cpu()
        neighbor_mean = (near_neighbor @ pseudo.unsqueeze(1)).squeeze(1)
        neighbor_mean = neighbor_mean / near_neighbor.sum(dim=1).clamp(min=1.0)
        mask = (neighbor_mean - pseudo).abs().le(0.3)
        return mask.to(self.device, dtype=torch.float32)

    def _adaptive_weight(self, corrected_entropy: torch.Tensor, pseudo_dist: torch.Tensor) -> torch.Tensor:
        entropy_exp_term = torch.exp(corrected_entropy - self.tau_ent)
        if self.confidence_metric_mode != "entropy":
            raise ValueError(f"unsupported confidence_metric_mode={self.confidence_metric_mode}")
        lhs_weight = 1.0 / entropy_exp_term
        if self.sample_selection_mode == "threshold":
            select_mask = (corrected_entropy < self.entropy_threshold).float()
        elif self.sample_selection_mode in {"none", "ftta_neighbors"}:
            select_mask = torch.ones_like(lhs_weight)
        else:
            raise ValueError(f"unsupported sample_selection_mode={self.sample_selection_mode}")

        if self.adaptive_weight_mode == "intended_product":
            if self.intended_product_nu_mode == "separate_multiplier":
                weight = lhs_weight * (self.nu / (1.0 + self.eta * pseudo_dist))
            elif self.intended_product_nu_mode == "denom_plus_nu":
                weight = (1.0 / (entropy_exp_term + self.nu)) * (1.0 / (1.0 + self.eta * pseudo_dist))
            else:
                raise ValueError(f"unsupported intended_product_nu_mode={self.intended_product_nu_mode}")
        elif self.adaptive_weight_mode == "current_sum":
            weight = lhs_weight + (self.nu / (1.0 + self.eta * pseudo_dist))
        else:
            raise ValueError(f"unsupported adaptive_weight_mode={self.adaptive_weight_mode}")
        return weight * select_mask

    def _batch_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features, probs = self._get_features_and_probs(batch)
        corrected_probs = self.corrector.process_probs(probs.detach(), update_pi=False)
        corrected_entropy = prob_entropy(corrected_probs).detach()
        distances, similarity = self._fca(features)
        distance_term = torch.exp(-distances / max(self.distance_eps, EPS))
        hybrid = (1.0 - self.alpha) * distance_term + self.alpha * corrected_probs
        positive_idx = torch.topk(hybrid, k=min(self.top_k, hybrid.shape[1]), dim=1).indices
        nc_loss = self._multi_positive_nce(similarity, positive_idx)
        ent_loss = prob_entropy(probs)
        pseudo_idx = corrected_probs.argmax(dim=1, keepdim=True)
        pseudo_dist = distances.gather(1, pseudo_idx).squeeze(1).detach()
        adaptive_weight = self._adaptive_weight(corrected_entropy, pseudo_dist)
        if self.sample_selection_mode == "ftta_neighbors":
            adaptive_weight = adaptive_weight * self._neighbor_mask(batch, probs)
        loss_vec = adaptive_weight * (ent_loss + nc_loss)
        return loss_vec.sum() / adaptive_weight.sum().clamp(min=EPS)

    @torch.enable_grad()
    def adapt_and_predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        for _ in range(self.steps_per_batch):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(batch)
            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            _, probs_after = self._get_features_and_probs(batch)
            corrected = self.corrector.process_probs(probs_after, update_pi=True)
        return corrected[:, 1].detach()


def run_nctta_entmix_eval(
    bundle: SeedBundle,
    batch_size: int,
    device: torch.device,
    cfg: NCTTAEntMixConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adapter = NCTTAEntMixAdapter(bundle.model, bundle.source_prior, cfg, device=device)
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for batch, y in build_batches(bundle.dataset, bundle.split, batch_size, device=device, shuffle=False):
        probs = adapter.adapt_and_predict(batch)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float32))
        y_all.append(apply_label_mapping(y.detach().cpu().numpy(), bundle.label_mapping))
    return (
        np.concatenate(probs_all),
        np.concatenate(y_all),
        adapter.corrector.pi_t.detach().cpu().numpy().astype(np.float32),
    )


def resolve_json_selected_params(payload: Mapping[str, Any], dataset: str, model: str) -> dict[str, Any]:
    method_block = dict(payload.get("nctta_entmix", payload.get("propose", {})))
    selected = dict(method_block.get("selected", {}))
    params: dict[str, Any] = {}
    default_params = selected.get("default")
    if isinstance(default_params, dict):
        params.update(default_params)
    datasets_block = selected.get("datasets", {})
    if isinstance(datasets_block, dict) and dataset in datasets_block:
        dataset_block = datasets_block[dataset]
        if isinstance(dataset_block, dict):
            dataset_default = dataset_block.get("default")
            if isinstance(dataset_default, dict):
                params.update(dataset_default)
            models_block = dataset_block.get("models", {})
            if isinstance(models_block, dict) and isinstance(models_block.get(model), dict):
                params.update(models_block[model])
    return params


def selected_hpo_params(dataset: str, model: str, selected_json: str | None = None) -> dict[str, Any]:
    if selected_json:
        payload = json.loads(Path(selected_json).read_text(encoding="utf-8"))
        params = resolve_json_selected_params(payload, dataset, model)
        if params:
            return params
    params = dict(COMMON_HPO_DEFAULTS)
    params.update(HPO_BACC_20260412[dataset][model])
    return params


def cfg_from_params(params: Mapping[str, Any]) -> NCTTAEntMixConfig:
    return NCTTAEntMixConfig(
        lr=float(params.get("nctta_lr", 1e-4)),
        weight_decay=float(params.get("nctta_weight_decay", 0.0)),
        steps_per_batch=int(params.get("nctta_steps_per_batch", 1)),
        alpha=float(params.get("nctta_alpha", 0.35)),
        top_k=int(params.get("nctta_top_k", 1)),
        distance_eps=float(params.get("nctta_distance_eps", 0.5)),
        tau_ent=float(params.get("nctta_tau_ent", 0.3)),
        entropy_threshold=None
        if params.get("nctta_entropy_threshold") is None
        else float(params.get("nctta_entropy_threshold")),
        nu=float(params.get("nctta_nu", 1.0)),
        eta=float(params.get("nctta_eta", 1.0)),
        adaptive_weight_mode=str(params.get("nctta_adaptive_weight_mode", "intended_product")),
        intended_product_nu_mode=str(params.get("nctta_intended_product_nu_mode", "separate_multiplier")),
        sample_selection_mode=str(params.get("nctta_sample_selection_mode", "threshold")),
        confidence_metric_mode=str(params.get("nctta_confidence_metric_mode", "entropy")),
        entmix=EntMixConfig(
            prior_alpha=float(params.get("entmix_prior_alpha", 0.95)),
            prior_beta=float(params.get("entmix_prior_beta", 1.0)),
            entmix_min=float(params.get("entmix_min", 0.0)),
            entmix_max=float(params.get("entmix_max", 1.0)),
            target_prior_init=str(params.get("entmix_target_prior_init", "uniform")),
        ),
    )


def resolve_batch_size(params: Mapping[str, Any], model_name: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return int(fallback)
    if "batch_size" in params:
        return int(params["batch_size"])
    batch_size_by_model = params.get("batch_size_by_model")
    if isinstance(batch_size_by_model, dict):
        if model_name in batch_size_by_model:
            return int(batch_size_by_model[model_name])
        if "default" in batch_size_by_model:
            return int(batch_size_by_model["default"])
    return 128 if model_name == "ft_transformer" else 512


def parse_methods(value: str) -> list[str]:
    aliases = {"baseline": "baseline", "unadapt": "baseline", "nctta_entmix": "nctta_entmix", "propose": "nctta_entmix"}
    methods = [aliases[x.strip().lower()] for x in str(value).split(",") if x.strip()]
    invalid = [x for x in methods if x not in {"baseline", "nctta_entmix"}]
    if invalid:
        raise ValueError(f"unsupported methods={invalid}")
    out: list[str] = []
    for method in methods:
        if method not in out:
            out.append(method)
    return out


def baseline_row(bundle: SeedBundle, metrics: Mapping[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": now_str(),
        "dataset": bundle.dataset_name,
        "dataset_display": display_name(bundle.dataset_name),
        "model": bundle.model_name,
        "model_display": display_name(bundle.model_name),
        "seed": int(bundle.seed),
        "split": bundle.split,
        "method": "baseline",
        "checkpoint_path": str(bundle.checkpoint_path),
    }
    for key in METRIC_ORDER:
        row[key] = float(metrics.get(key, float("nan")))
    return row


def adapted_row(
    bundle: SeedBundle,
    baseline_metrics: Mapping[str, float],
    adapted_metrics: Mapping[str, float],
    final_prior: Sequence[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": now_str(),
        "dataset": bundle.dataset_name,
        "dataset_display": display_name(bundle.dataset_name),
        "model": bundle.model_name,
        "model_display": display_name(bundle.model_name),
        "seed": int(bundle.seed),
        "split": bundle.split,
        "method": "nctta_entmix",
        "checkpoint_path": str(bundle.checkpoint_path),
    }
    for key in METRIC_ORDER:
        row[f"baseline_{key}"] = float(baseline_metrics.get(key, float("nan")))
        row[f"adapted_{key}"] = float(adapted_metrics.get(key, float("nan")))
    row["source_prior_0"] = float(bundle.source_prior[0].item())
    row["source_prior_1"] = float(bundle.source_prior[1].item())
    row["final_prior_0"] = float(final_prior[0]) if len(final_prior) > 0 else float("nan")
    row["final_prior_1"] = float(final_prior[1]) if len(final_prior) > 1 else float("nan")
    return row


def save_predictions(results_root: Path, dataset: str, method: str, model: str, seed: int, y_prob: np.ndarray, y_true: np.ndarray) -> None:
    out_dir = results_root / dataset / method / "predictions" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"seed{seed}.npz", y_prob=y_prob.astype(np.float32), y_true=y_true.astype(np.int64))


def write_overall_compact(results_root: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(results_root.glob("*/*/summary_compact_runs.csv")):
        rows.extend(read_rows(path))
    if rows:
        write_rows(results_root / "all_summary_compact_runs.csv", rows, ["dataset", "method", "model", "seed", *METRIC_ORDER])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline and NCTTA+EntMix on pretrained TabReD checkpoints.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--methods", default="baseline,nctta_entmix")
    parser.add_argument("--tabred_root", default=None)
    parser.add_argument("--artifacts_root", default=None)
    parser.add_argument("--pretrain_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--selected_json", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> int:
    configure_tabred(args.tabred_root)
    artifacts = resolve_artifacts_root(args.artifacts_root)
    pretrain_root = Path(args.pretrain_root).resolve() if args.pretrain_root else artifacts / "pretrain"
    results_root = Path(args.results_root).resolve() if args.results_root else artifacts / "results"
    models = resolve_models(args.models)
    datasets = resolve_datasets(args.datasets)
    seeds = parse_int_csv_arg(args.seeds)
    methods = parse_methods(args.methods)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    failures: list[tuple[str, str, int, str]] = []
    tasks = [(dataset_name, model_name, seed) for dataset_name in datasets for model_name in models for seed in seeds]

    for task_idx, (dataset_name, model_name, seed) in enumerate(tasks):
        if task_idx % args.num_shards != args.shard_index:
            continue
        params = selected_hpo_params(dataset_name, model_name, args.selected_json)
        batch_size = resolve_batch_size(params, model_name, args.batch_size)
        print(
            f"[tta] shard={args.shard_index}/{args.num_shards} task={task_idx} "
            f"dataset={dataset_name} model={model_name} seed={seed} methods={methods} "
            f"batch_size={batch_size} device={device}"
        )
        if args.dry_run:
            continue

        try:
            set_seed(seed)
            bundle = prepare_seed_bundle(
                pretrain_root=pretrain_root,
                dataset_name=dataset_name,
                model_name=model_name,
                seed=seed,
                device=device,
                split=args.split,
            )
            baseline_prob, y_true = run_baseline_eval(bundle, batch_size, device=device)
            baseline_metrics = compute_binary_metrics(y_true, baseline_prob)

            if "baseline" in methods:
                summary = results_root / dataset_name / "baseline" / "summary.csv"
                compact = results_root / dataset_name / "baseline" / "summary_compact_runs.csv"
                stats = results_root / dataset_name / "baseline" / "summary_stats.csv"
                upsert_csv(summary, baseline_row(bundle, baseline_metrics), tta_row_header(include_adapted=False), ["dataset", "method", "model", "seed"])
                upsert_csv(compact, build_compact_row(dataset_name, "baseline", model_name, seed, baseline_metrics), tta_compact_run_header(), ["dataset", "method", "model", "seed"])
                update_group_stats(summary, stats, "model")
                save_predictions(results_root, dataset_name, "baseline", model_name, seed, baseline_prob, y_true)

            if "nctta_entmix" in methods:
                cfg = cfg_from_params(params)
                adapted_prob, adapted_y_true, final_prior = run_nctta_entmix_eval(bundle, batch_size, device=device, cfg=cfg)
                adapted_metrics = compute_binary_metrics(adapted_y_true, adapted_prob)
                summary = results_root / dataset_name / "nctta_entmix" / "summary.csv"
                compact = results_root / dataset_name / "nctta_entmix" / "summary_compact_runs.csv"
                stats = results_root / dataset_name / "nctta_entmix" / "summary_stats.csv"
                upsert_csv(summary, adapted_row(bundle, baseline_metrics, adapted_metrics, final_prior), tta_row_header(include_adapted=True), ["dataset", "method", "model", "seed"])
                upsert_csv(compact, build_compact_row(dataset_name, "nctta_entmix", model_name, seed, adapted_metrics), tta_compact_run_header(), ["dataset", "method", "model", "seed"])
                update_group_stats(summary, stats, "model")
                save_predictions(results_root, dataset_name, "nctta_entmix", model_name, seed, adapted_prob, adapted_y_true)

            write_dataset_method_summary(results_root, dataset_name)
            write_overall_compact(results_root)
        except Exception as exc:  # noqa: BLE001
            failures.append((dataset_name, model_name, seed, str(exc)))
            print(f"[tta] FAILED dataset={dataset_name} model={model_name} seed={seed} err={exc}")
            if not args.continue_on_error:
                break

    print(f"[tta] results={results_root}")
    if failures:
        print("[tta] failures")
        for dataset_name, model_name, seed, err in failures:
            print(f"  - dataset={dataset_name} model={model_name} seed={seed}: {err}")
        return 1
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
