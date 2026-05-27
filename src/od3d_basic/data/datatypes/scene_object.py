from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
from od3d_basic.data.datatypes.frame import _stack_field
from od3d_basic.data.datatypes.scene import Scene
from od3d_basic.data.datatypes.object import Object


@dataclass(kw_only=True)
class SceneObject(Scene, Object):
    scene_object_id:   str
    cams_bbox2d:       Optional[Tensor] = None  # (T, 4)
    cams_bbox3d:       Optional[Tensor] = None  # (T, 8, 3)
    so_masks:          Optional[Tensor] = None  # (T, H, W)  bool  object-instance masks
    cams_tform4x4_obj: Optional[Tensor] = None  # (T, 4, 4)

    @staticmethod
    def from_frame_objects(
        frame_objects: list,  # list[FrameObject]
        scene_object_id: str = "",
        scene_id: str = "",
    ) -> SceneObject:
        from od3d_basic.data.datatypes.frame_object import FrameObject

        def _stack(attr):
            vals = [getattr(fo, attr) for fo in frame_objects]
            return torch.stack(vals) if all(v is not None for v in vals) else None

        def _stack_lvls(attr):
            per_fo = [getattr(fo, attr) for fo in frame_objects]
            if any(v is None for v in per_fo):
                return None
            return [
                torch.stack([per_fo[t][l] for t in range(len(per_fo))])
                for l in range(len(per_fo[0]))
            ]

        fo0 = frame_objects[0]
        return SceneObject(
            scene_object_id = scene_object_id,
            scene_id        = scene_id or fo0.frame_id,
            object_id       = fo0.object_id,
            # Scene fields (stacked over T frames)
            cams_intr4x4  = _stack("cam_intr4x4"),
            rgbs          = _stack("rgb"),
            depths        = _stack("depth"),
            depths_masks  = _stack("depth_mask"),
            masks         = _stack("mask"),
            feats         = _stack("feat"),
            featmaps      = _stack("featmap"),
            featmaps_lvls = _stack_lvls("featmap_lvls"),
            frames        = list(frame_objects),
            # SceneObject fields (stacked over T frames)
            cams_bbox2d       = _stack("cam_bbox2d"),
            cams_bbox3d       = _stack("cam_bbox3d"),
            so_masks          = _stack("fo_mask"),
            cams_tform4x4_obj = _stack("cam_tform4x4_obj"),
            # Object fields (time-invariant, taken from first frame)
            pts3d                   = fo0.pts3d,
            pts3d_feats             = fo0.pts3d_feats,
            pts3d_feats_mask        = fo0.pts3d_feats_mask,
            verts3d_feats           = fo0.verts3d_feats,
            verts3d_feats_mask      = fo0.verts3d_feats_mask,
            mesh                    = fo0.mesh,
            obj_ncds0c_tform4x4_obj = fo0.obj_ncds0c_tform4x4_obj,
            obj_kpts3d              = fo0.obj_kpts3d,
            obj_kpts3d_mask         = fo0.obj_kpts3d_mask,
            category                = fo0.category,
            attributes              = fo0.attributes,
        )


@dataclass
class SceneObjectBatch:
    """Stacked across B samples. Each sample = 1 scene (T frames) + 1 object."""
    # scene
    cams_intr4x4:  Optional[Tensor]       = None  # (B, T, 4, 4)
    rgbs:          Optional[Tensor]       = None  # (B, T, 3, H, W)
    depths:        Optional[Tensor]       = None  # (B, T, H, W)
    depths_masks:  Optional[Tensor]       = None  # (B, T, H, W)
    scene_masks:   Optional[Tensor]       = None  # (B, T, H, W)
    feats:         Optional[Tensor]       = None  # (B, T, F)
    featmaps:      Optional[Tensor]       = None  # (B, T, F, H, W)
    featmaps_lvls: Optional[List[Tensor]] = None  # L x (B, T, F, H_l, W_l)
    # scene-object
    cams_bbox2d:       Optional[Tensor]   = None  # (B, T, 4)
    cams_bbox3d:       Optional[Tensor]   = None  # (B, T, 8, 3)
    so_masks:          Optional[Tensor]   = None  # (B, T, H, W)
    cams_tform4x4_obj: Optional[Tensor]   = None  # (B, T, 4, 4)
    # object (time-invariant)
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[Tensor] = None  # (B,)  int64


def collate_scene_objects(
    samples: list[SceneObject],
    include: Optional[set[str]] = None,
) -> SceneObjectBatch:
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

    return SceneObjectBatch(
        cams_intr4x4  = _get("cams_intr4x4"),
        rgbs          = _get("rgbs"),
        depths        = _get("depths"),
        depths_masks  = _get("depths_masks"),
        scene_masks   = _get("masks"),
        feats         = _get("feats"),
        featmaps      = _get("featmaps"),
        featmaps_lvls = _get_lvls("featmaps_lvls"),
        cams_bbox2d       = _get("cams_bbox2d"),
        cams_bbox3d       = _get("cams_bbox3d"),
        so_masks          = _get("so_masks"),
        cams_tform4x4_obj = _get("cams_tform4x4_obj"),
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
