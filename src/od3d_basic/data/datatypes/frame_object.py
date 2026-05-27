from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
from od3d_basic.data.datatypes.mesh import Mesh
from od3d_basic.data.datatypes.frame import Frame, _stack_field
from od3d_basic.data.datatypes.object import Object


@dataclass(kw_only=True)
class FrameObject(Frame, Object):
    frame_object_id:  str
    cam_bbox2d:       Optional[Tensor] = None  # (4,)     xyxy pixels
    cam_bbox3d:       Optional[Tensor] = None  # (8, 3)   3-D corners
    fo_mask:          Optional[Tensor] = None  # (H, W)   bool  object-instance mask
    cam_tform4x4_obj: Optional[Tensor] = None  # (4, 4)  cam←obj SE(3)


@dataclass
class FrameObjectBatch:
    """Stacked across B samples. Each sample = 1 frame + 1 object."""
    # frame
    cam_intr4x4:      Optional[Tensor]       = None  # (B, 4, 4)
    rgb:              Optional[Tensor]       = None  # (B, 3, H, W)
    depth:            Optional[Tensor]       = None  # (B, H, W)
    depth_mask:       Optional[Tensor]       = None  # (B, H, W)
    frame_mask:       Optional[Tensor]       = None  # (B, H, W)
    feat:             Optional[Tensor]       = None  # (B, F)
    featmap:          Optional[Tensor]       = None  # (B, F, H, W)
    featmap_lvls:     Optional[List[Tensor]] = None  # L x (B, F, H_l, W_l)
    # frame-object
    cam_bbox2d:       Optional[Tensor]       = None  # (B, 4)
    cam_bbox3d:       Optional[Tensor]       = None  # (B, 8, 3)
    fo_mask:          Optional[Tensor]       = None  # (B, H, W)
    cam_tform4x4_obj: Optional[Tensor]       = None  # (B, 4, 4)
    # object
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[Tensor] = None  # (B,)  int64
    mesh:                    Optional[Mesh]   = None  # shared mesh for all B viewpoints


def collate_frame_objects(
    samples: list[FrameObject],
    include: Optional[set[str]] = None,
) -> FrameObjectBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, attr) for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return FrameObjectBatch(
        cam_intr4x4      = _get("cam_intr4x4"),
        rgb              = _get("rgb"),
        depth            = _get("depth"),
        depth_mask       = _get("depth_mask"),
        frame_mask       = _get("mask"),
        feat             = _get("feat"),
        featmap          = _get("featmap"),
        featmap_lvls     = _get_lvls("featmap_lvls"),
        cam_bbox2d       = _get("cam_bbox2d"),
        cam_bbox3d       = _get("cam_bbox3d"),
        fo_mask          = _get("fo_mask"),
        cam_tform4x4_obj = _get("cam_tform4x4_obj"),
        pts3d                   = _get("pts3d"),
        pts3d_feats             = _get("pts3d_feats"),
        pts3d_feats_mask        = _get("pts3d_feats_mask"),
        verts3d_feats           = _get("verts3d_feats"),
        verts3d_feats_mask      = _get("verts3d_feats_mask"),
        obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj"),
        obj_kpts3d              = _get("obj_kpts3d"),
        obj_kpts3d_mask         = _get("obj_kpts3d_mask"),
        category = _stack_field([
            torch.tensor(s.category) if s.category is not None else None
            for s in samples
        ]) if (include is None or "category" in include) else None,
    )
