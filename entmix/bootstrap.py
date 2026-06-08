from __future__ import annotations

import argparse

from .paths import TABRED_COMMIT, TABRED_REPO_URL
from .tabred import clone_or_checkout_tabred, ensure_tabred_data_link


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone clean upstream TabReD and wire Entmix data storage.")
    parser.add_argument("--tabred_root", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--repo_url", default=TABRED_REPO_URL)
    parser.add_argument("--commit", default=TABRED_COMMIT)
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tabred_root = clone_or_checkout_tabred(
        tabred_root=args.tabred_root,
        repo_url=args.repo_url,
        commit=args.commit,
        force=args.force,
    )
    data_link = ensure_tabred_data_link(tabred_root=tabred_root, data_root=args.data_root)
    print(f"[bootstrap] TabReD root: {tabred_root}")
    print(f"[bootstrap] TabReD commit: {args.commit}")
    print(f"[bootstrap] TabReD data path: {data_link} -> {data_link.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
