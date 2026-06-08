from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABRED_ROOT = PROJECT_ROOT / "tabred"
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

TABRED_REPO_URL = "https://github.com/yandex-research/tabred.git"
TABRED_COMMIT = "36352fc567f5fb396bfc55bdec04e3cdf923e941"


def resolve_tabred_root(value: str | os.PathLike[str] | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("ENTMIX_TABRED_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_TABRED_ROOT.resolve()


def resolve_artifacts_root(value: str | os.PathLike[str] | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return ARTIFACTS_ROOT.resolve()


def resolve_data_root(value: str | os.PathLike[str] | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return DATA_ROOT.resolve()


def ensure_project_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
