"""
od3d_basic dataset CLI.

Usage:
  od3d_dataset fetch     --config housecorr3d.yaml [--url URL] [--platform PLATFORM]
  od3d_dataset index     --config housecorr3d.yaml [--db index.db] [--platform PLATFORM]
  od3d_dataset viz       --config housecorr3d.yaml [--db index.db] [--limit N] [--object-id ID] [--render] [--platform PLATFORM]

The config YAML must contain a 'class_name' field that matches a registered
dataset (e.g. 'HouseCorr3D').  All paths and options are read from the config;
command-line flags only override what needs to be overridden at run-time.
When --platform is given the platform's path_datasets_raw / path_datasets_preprocess
values override the corresponding variables in the dataset config before Hydra
resolves any ${} interpolations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from od3d_basic.dataset.dataset import DatasetConfig, _REGISTRY_DATASETS, _ensure_dataset_imported


def _platform_to_dataset_overrides(platform: str) -> list[str]:
    """Return Hydra override strings derived from the platform config."""
    from od3d_basic.cli import _load_platform_config
    plat_cfg, _ = _load_platform_config(platform)
    overrides: list[str] = []
    if raw := plat_cfg.get("path_datasets_raw"):
        overrides.append(f"path_datasets_raw={raw}")
    if pre := plat_cfg.get("path_datasets_preprocess"):
        overrides.append(f"path_datasets_preprocess={pre}")
    return overrides


def _load_class_from_config(config_path: Path, overrides: list[str] | None = None):
    cfg = DatasetConfig.from_yaml(config_path, overrides=overrides)
    _ensure_dataset_imported(cfg.class_name)

    lower = cfg.class_name.lower()
    for key, cls in _REGISTRY_DATASETS.items():
        if key.lower() == lower:
            return cls, cfg

    known = sorted(_REGISTRY_DATASETS)
    print(
        f"ERROR: class_name '{cfg.class_name}' from {config_path} is not registered.\n"
        f"Known datasets: {known if known else '(none — is the subpackage installed?)'}",
        file=sys.stderr,
    )
    sys.exit(1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="od3d_dataset",
        description="od3d dataset CLI — fetch, index, and visualise any registered dataset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_config(p):
        p.add_argument(
            "--config", required=True, type=Path, metavar="YAML",
            help="Path to a DatasetConfig YAML file (must contain class_name)",
        )
        p.add_argument(
            "--platform", default="default", metavar="PLATFORM",
            help="Platform name whose path_datasets_raw / path_datasets_preprocess "
                 "override the dataset config paths (default: default)",
        )

    p_fetch = sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL", help="ZIP download URL")

    p_index = sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite output file (default: <path_preprocess>/index.db)",
    )

    p_vis = sub.add_parser("viz", help="Show dataset summary and optionally render meshes")
    _add_config(p_vis)
    p_vis.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite index file (default: <path_preprocess>/index.db)",
    )
    p_vis.add_argument("--limit", type=int, default=20, metavar="N",
                       help="Max objects to show (default: 20)")
    p_vis.add_argument("--object-id", default=None, metavar="ID",
                       help="Show only this object")
    p_vis.add_argument("--filter-has-kpts", action="store_true",
                       help="Only show objects that have obj_kpts3d in the index")
    p_vis.add_argument("--render", action="store_true",
                       help="Open interactive viser viewer")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N",
                       help="Number of viewpoints to render when --render is set (default: 4)")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender",
                       help="Renderer backend for --render-frames (default: pyrender)")

    args = parser.parse_args(argv)
    overrides = _platform_to_dataset_overrides(args.platform)
    cls, cfg = _load_class_from_config(args.config, overrides=overrides)

    if args.command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.command == "index":
        cls.index(cfg, db=args.db)
    elif args.command == "viz":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
        )


if __name__ == "__main__":
    main()
