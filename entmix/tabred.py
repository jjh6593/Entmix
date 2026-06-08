from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import DATA_ROOT, TABRED_COMMIT, TABRED_REPO_URL, ensure_project_dirs, resolve_tabred_root


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(args))
    subprocess.run(args, cwd=None if cwd is None else str(cwd), check=True)


def clone_or_checkout_tabred(
    tabred_root: str | os.PathLike[str] | None = None,
    repo_url: str = TABRED_REPO_URL,
    commit: str = TABRED_COMMIT,
    force: bool = False,
) -> Path:
    ensure_project_dirs()
    root = resolve_tabred_root(tabred_root)

    if root.exists() and force:
        shutil.rmtree(root)

    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", repo_url, str(root)])
    elif not (root / ".git").exists():
        raise RuntimeError(f"{root} exists but is not a git checkout. Move it or pass --force.")

    run_command(["git", "fetch", "--tags", "origin"], cwd=root)
    run_command(["git", "checkout", commit], cwd=root)
    return root


def ensure_tabred_data_link(
    tabred_root: str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    root = resolve_tabred_root(tabred_root)
    data_dir = Path(data_root).expanduser().resolve() if data_root else DATA_ROOT.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    link_path = root / "data"
    if link_path.is_symlink():
        current = link_path.resolve()
        if current != data_dir:
            link_path.unlink()
            link_path.symlink_to(os.path.relpath(data_dir, root))
    elif link_path.exists():
        if any(link_path.iterdir()):
            print(f"[bootstrap] keeping existing non-empty TabReD data directory: {link_path}")
            return link_path
        link_path.rmdir()
        link_path.symlink_to(os.path.relpath(data_dir, root))
    else:
        link_path.symlink_to(os.path.relpath(data_dir, root))

    return link_path


def add_tabred_to_path(tabred_root: str | os.PathLike[str] | None = None) -> Path:
    root = resolve_tabred_root(tabred_root)
    if not root.exists():
        raise FileNotFoundError(
            f"TabReD checkout not found at {root}. Run `python scripts/bootstrap_tabred.py` first."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def assert_tabred_ready(tabred_root: str | os.PathLike[str] | None = None) -> Path:
    root = add_tabred_to_path(tabred_root)
    required = [root / "lib", root / "bin" / "nn_baselines.py", root / "exp"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TabReD checkout is incomplete: {missing}")
    return root
