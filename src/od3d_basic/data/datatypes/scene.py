from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch import Tensor
from od3d_basic.data.datatypes.frame import Frame


@dataclass(kw_only=True)
class Scene:
    scene_id:      str
    cams_intr4x4:  Optional[Tensor]       = None  # (T, 4, 4)
    rgbs:          Optional[Tensor]       = None  # (T, H, W, 3)
    depths:        Optional[Tensor]       = None  # (T, H, W)
    depths_masks:  Optional[Tensor]       = None  # (T, H, W)
    masks:         Optional[Tensor]       = None  # (T, H, W)  bool
    feats:         Optional[Tensor]       = None  # (T, F)
    featmaps:      Optional[Tensor]       = None  # (T, H, W, F)
    featmaps_lvls: Optional[List[Tensor]] = None  # L x (T, H_l, W_l, F)
    frames:        list[Frame]            = field(default_factory=list)

    @staticmethod
    def from_frames(frames: list[Frame], scene_id: str = "") -> Scene:
        def _stack(attr):
            vals = [getattr(f, attr) for f in frames]
            return torch.stack(vals) if all(v is not None for v in vals) else None

        def _stack_lvls(attr):
            per_frame = [getattr(f, attr) for f in frames]
            if any(v is None for v in per_frame):
                return None
            return [
                torch.stack([per_frame[t][l] for t in range(len(per_frame))])
                for l in range(len(per_frame[0]))
            ]

        return Scene(
            scene_id      = scene_id,
            cams_intr4x4  = _stack("cam_intr4x4"),
            rgbs          = _stack("rgb"),
            depths        = _stack("depth"),
            depths_masks  = _stack("depth_mask"),
            masks         = _stack("mask"),
            feats         = _stack("feat"),
            featmaps      = _stack("featmap"),
            featmaps_lvls = _stack_lvls("featmap_lvls"),
            frames        = frames,
        )
