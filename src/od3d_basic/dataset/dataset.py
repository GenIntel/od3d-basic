from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import yaml

import torch
from torch.utils.data import Dataset as _TorchDataset, DataLoader
from od3d_basic.data.modalities import (
    FrameObject, SceneObject,
    FrameObjectBatch, SceneObjectBatch,
    ObjectPair,
)


# ── Item / batch type enums ───────────────────────────────────────────────────

class ItemType(str, Enum):
    OBJECT       = "object"
    OBJECT_PAIR  = "object_pair"
    FRAME_OBJECT = "frame_object"
    SCENE_OBJECT = "scene_object"

class BatchType(str, Enum):
    OBJECT       = "object"
    OBJECT_PAIR  = "object_pair"
    FRAME_OBJECT = "frame_object"
    SCENE_OBJECT = "scene_object"


# ── Dataset config ────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # what and where
    class_name:      str                   # e.g. "HouseCorr3D"
    root:            Path                  = Path("data")
    path_raw:        Optional[Path]        = None   # raw (downloaded) data root
    path_preprocess: Optional[Path]        = None   # preprocessed data root
    split:           str                   = "train"

    # item / batch shape
    item_type:     ItemType                = ItemType.OBJECT
    batch_type:    BatchType               = BatchType.FRAME_OBJECT
    scene_length:  int                     = 8      # T; ignored for OBJECT / FRAME_OBJECT

    # mesh variant: "default" uses the raw mesh path; others load/create a
    # marching-cubes remesh at path_preprocess/mesh/<mesh_type>/<obj_id>.glb
    mesh_type:     str                     = "default"

    # which modalities to load (None = all available)
    modalities:        Optional[set[str]]  = None
    object_modalities: Optional[set[str]]  = None

    # filtering
    categories:      Optional[list[str]]   = None   # None = all
    max_samples:     Optional[int]         = None
    filter_has_kpts: bool                  = False

    # extra per-dataset kwargs passed through to the implementation
    extra: dict                            = field(default_factory=dict)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_yaml(self, path: Path) -> None:
        d = {
            "class_name":      self.class_name,
            "root":            str(self.root),
            "path_raw":        str(self.path_raw) if self.path_raw else None,
            "path_preprocess": str(self.path_preprocess) if self.path_preprocess else None,
            "split":           self.split,
            "item_type":       self.item_type.value,
            "batch_type":      self.batch_type.value,
            "scene_length":    self.scene_length,
            "mesh_type":       self.mesh_type,
            "modalities":        sorted(self.modalities)        if self.modalities        else None,
            "object_modalities": sorted(self.object_modalities) if self.object_modalities else None,
            "categories":      self.categories,
            "max_samples":     self.max_samples,
            "filter_has_kpts": self.filter_has_kpts,
            "extra":           self.extra,
        }
        path.write_text(yaml.safe_dump(d, sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            class_name      = d["class_name"],
            root            = Path(d["root"]) if d.get("root") else Path("data"),
            path_raw        = Path(d["path_raw"]) if d.get("path_raw") else None,
            path_preprocess = Path(d["path_preprocess"]) if d.get("path_preprocess") else None,
            split           = d.get("split", "train"),
            item_type       = ItemType(d.get("item_type", "object")),
            batch_type      = BatchType(d.get("batch_type", "frame_object")),
            scene_length    = d.get("scene_length", 8),
            mesh_type       = d.get("mesh_type", "default"),
            modalities        = set(d["modalities"])        if d.get("modalities")        else None,
            object_modalities = set(d["object_modalities"]) if d.get("object_modalities") else None,
            categories      = d.get("categories"),
            max_samples     = d.get("max_samples"),
            filter_has_kpts = bool(d.get("filter_has_kpts", False)),
            extra           = d.get("extra", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path, overrides: "list[str] | None" = None) -> "DatasetConfig":
        return cls.from_dict(_load_yaml_with_defaults(Path(path), overrides=overrides))


def _load_yaml_with_defaults(path: Path, overrides: "list[str] | None" = None) -> dict:
    """Load a YAML config via Hydra compose, resolving defaults and ${} interpolations."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.stem, overrides=overrides or [])
    GlobalHydra.instance().clear()
    return OmegaConf.to_container(cfg, resolve=True)


# ── Dataset registry ──────────────────────────────────────────────────────────

_REGISTRY_DATASETS: dict[str, type["ConfigurableDataset"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "HouseCorr3D": "od3d_basic.dataset.housecorr3d.dataset",
}


def _ensure_dataset_imported(name: str) -> None:
    if name not in _REGISTRY_DATASETS and name in _CLASS_TO_MODULE:
        import importlib
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_dataset(name: str):
    """Class decorator: @register_dataset("HouseCorr3D")"""
    def decorator(cls):
        _REGISTRY_DATASETS[name] = cls
        return cls
    return decorator


def build_dataset(cfg: DatasetConfig) -> "ConfigurableDataset":
    _ensure_dataset_imported(cfg.class_name)
    if cfg.class_name not in _REGISTRY_DATASETS:
        raise KeyError(
            f"Unknown dataset '{cfg.class_name}'. "
            f"Registered: {sorted(_REGISTRY_DATASETS)}"
        )
    return _REGISTRY_DATASETS[cfg.class_name](cfg)


# ── Base class ────────────────────────────────────────────────────────────────

class ConfigurableDataset(_TorchDataset):
    """
    Subclass this and implement _load_frame_object / _load_scene_object.
    Everything else — collation, DataLoader wiring — is handled here.
    """

    categories: tuple[str, ...] = ()  # override in subclasses with dataset-specific names

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._index: list = []
        self._setup()

    def _setup(self) -> None:
        pass

    # ── item loading (implement the one(s) matching your item_type) ──────────

    def _load_object(self, idx: int):
        raise NotImplementedError

    def _load_object_pair(self, idx: int) -> ObjectPair:
        raise NotImplementedError

    def _load_frame_object(self, idx: int) -> FrameObject:
        raise NotImplementedError

    def _load_scene_object(self, idx: int) -> SceneObject:
        raise NotImplementedError

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        if self.cfg.item_type == ItemType.OBJECT:
            return self._load_object(idx)
        elif self.cfg.item_type == ItemType.OBJECT_PAIR:
            return self._load_object_pair(idx)
        elif self.cfg.item_type == ItemType.FRAME_OBJECT:
            return self._load_frame_object(idx)
        else:
            return self._load_scene_object(idx)

    # ── Collation ─────────────────────────────────────────────────────────────

    def collate_fn(
        self, samples: list[Union[FrameObject, SceneObject]]
    ) -> Union[FrameObjectBatch, SceneObjectBatch]:
        if self.cfg.batch_type == BatchType.FRAME_OBJECT:
            return collate_frame_objects(samples, include=self.cfg.modalities)
        else:
            return collate_scene_objects(samples, include=self.cfg.modalities)

    # ── DataLoader factory ────────────────────────────────────────────────────

    def build_loader(self, batch_size: int = 8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    # ── Dataset-level CLI hooks (override in subclasses) ──────────────────────

    @classmethod
    def fetch(cls, cfg: "DatasetConfig", *, url: Optional[str] = None) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement fetch()")

    @classmethod
    def index(cls, cfg: "DatasetConfig", *, db: Optional[Path] = None) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement index()")

    @classmethod
    def visualize(
        cls,
        cfg: "DatasetConfig",
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
    ) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement visualize()")
