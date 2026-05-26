from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Optional
import torch
from torch import Tensor


# ── Modality dataclasses ──────────────────────────────────────────────────────

@dataclass
class FrameModalities:
    cam_intr4x4:  Optional[Tensor]       = None  # (4, 4)
    rgb:          Optional[Tensor]       = None  # (H, W, 3)
    depth:        Optional[Tensor]       = None  # (H, W)
    depth_mask:   Optional[Tensor]       = None  # (H, W)  bool
    mask:         Optional[Tensor]       = None  # (H, W)  bool
    feat:         Optional[Tensor]       = None  # (F,)
    featmap:      Optional[Tensor]       = None  # (H, W, F)
    featmap_lvls: Optional[List[Tensor]] = None  # L x (H_l, W_l, F)


@dataclass
class SceneModalities:
    """Temporal stack of FrameModalities — shapes gain a leading T dim."""
    cams_intr4x4:  Optional[Tensor]       = None  # (T, 4, 4)
    rgbs:          Optional[Tensor]       = None  # (T, H, W, 3)
    depths:        Optional[Tensor]       = None  # (T, H, W)
    depths_masks:  Optional[Tensor]       = None  # (T, H, W)  bool
    masks:         Optional[Tensor]       = None  # (T, H, W)  bool
    feats:         Optional[Tensor]       = None  # (T, F)
    featmaps:      Optional[Tensor]       = None  # (T, H, W, F)
    featmaps_lvls: Optional[List[Tensor]] = None  # L x (T, H_l, W_l, F)

    @staticmethod
    def from_frames(frame_mods: list[FrameModalities]) -> "SceneModalities":
        """Stack a list of FrameModalities along a new time dimension."""
        def _stack(attr):
            vals = [getattr(f, attr) for f in frame_mods]
            return torch.stack(vals) if all(v is not None for v in vals) else None

        def _stack_lvls(attr):
            per_frame = [getattr(f, attr) for f in frame_mods]
            if any(v is None for v in per_frame):
                return None
            return [
                torch.stack([per_frame[t][l] for t in range(len(per_frame))])
                for l in range(len(per_frame[0]))
            ]

        return SceneModalities(
            cams_intr4x4  = _stack("cam_intr4x4"),
            rgbs          = _stack("rgb"),
            depths        = _stack("depth"),
            depths_masks  = _stack("depth_mask"),
            masks         = _stack("mask"),
            feats         = _stack("feat"),
            featmaps      = _stack("featmap"),
            featmaps_lvls = _stack_lvls("featmap_lvls"),
        )


@dataclass
class FrameObjectModalities:
    cam_bbox2d:       Optional[Tensor] = None  # (4,)     xyxy pixels
    cam_bbox3d:       Optional[Tensor] = None  # (8, 3)   3-D corners
    mask:             Optional[Tensor] = None  # (H, W)   bool
    cam_tform4x4_obj: Optional[Tensor] = None  # (4, 4)  cam←obj SE(3)


@dataclass
class SceneObjectModalities:
    """Temporal stack of FrameObjectModalities."""
    cams_bbox2d:       Optional[Tensor] = None  # (T, 4)
    cams_bbox3d:       Optional[Tensor] = None  # (T, 8, 3)
    masks:             Optional[Tensor] = None  # (T, H, W)
    cams_tform4x4_obj: Optional[Tensor] = None  # (T, 4, 4)

    @staticmethod
    def from_frame_objects(fo_mods: list[FrameObjectModalities]) -> "SceneObjectModalities":
        def _stack(attr):
            vals = [getattr(fo, attr) for fo in fo_mods]
            return torch.stack(vals) if all(v is not None for v in vals) else None

        return SceneObjectModalities(
            cams_bbox2d       = _stack("cam_bbox2d"),
            cams_bbox3d       = _stack("cam_bbox3d"),
            masks             = _stack("mask"),
            cams_tform4x4_obj = _stack("cam_tform4x4_obj"),
        )


@dataclass
class ObjectModalities:
    """Time-invariant — shared by both FrameObject and SceneObject samples."""
    pts3d:              Optional[Tensor] = None  # (N, 3)
    pts3d_feats:        Optional[Tensor] = None  # (N, F) or (N, V, F) multi-view
    pts3d_feats_mask:   Optional[Tensor] = None  # (N,) or (N, V) bool — valid (non-padded) entries
    verts3d_feats:      Optional[Tensor] = None  # (N, F) or (N, V, F) multi-view
    verts3d_feats_mask: Optional[Tensor] = None  # (N,) or (N, V) bool — valid (non-padded) entries
    mesh:               Optional[Mesh]   = None  # see below
    obj_ncds0c_tform4x4_obj:      Optional[Tensor] = None  # (4, 4)  normalized→original  (s*I | center; 0 | 1)
    obj_kpts3d:         Optional[Tensor] = None  # (K, 3)  object keypoints in object frame
    obj_kpts3d_mask:    Optional[Tensor] = None  # (K,)    bool — valid keypoints
    category:           Optional[int]    = None
    attributes:         Optional[dict]   = None


@dataclass
class Mesh:
    verts:       Tensor           # (V, 3)
    faces:       Tensor           # (F, 3)  int64
    vert_colors: Optional[Tensor] = None  # (V, 3) float RGB  — set when vertex colors available
    verts_uvs:   Optional[Tensor] = None  # (V, 2) float UV   — y-flipped to torch/image convention
    faces_uvs:   Optional[Tensor] = None  # (F, 3) int64 UV face indices
    texture:     Optional[Tensor] = None  # (3, H, W) float   — texture map


# ── Entity types ──────────────────────────────────────────────────────────────

@dataclass
class Frame:
    frame_id:   str
    modalities: FrameModalities = field(default_factory=FrameModalities)


@dataclass
class Object:
    object_id:  str
    modalities: ObjectModalities = field(default_factory=ObjectModalities)

@dataclass
class FrameObject:
    frame_object_id: str
    frame_id:        str
    object_id:       str
    frame_modalities:        FrameModalities       = field(default_factory=FrameModalities)
    frame_object_modalities: FrameObjectModalities = field(default_factory=FrameObjectModalities)
    object_modalities:       ObjectModalities      = field(default_factory=ObjectModalities)


@dataclass
class Scene:
    scene_id:   str
    modalities: SceneModalities = field(default_factory=SceneModalities)
    frames:     list[Frame]     = field(default_factory=list)


@dataclass
class SceneObject:
    scene_object_id: str
    scene_id:        str
    object_id:       str
    scene_modalities:        SceneModalities       = field(default_factory=SceneModalities)
    scene_object_modalities: SceneObjectModalities = field(default_factory=SceneObjectModalities)
    object_modalities:       ObjectModalities      = field(default_factory=ObjectModalities)


# ── Batch types ───────────────────────────────────────────────────────────────

@dataclass
class FrameObjectBatchModalities:
    """Stacked across B samples. Each sample = 1 frame + 1 object."""
    # frame
    cam_intr4x4:      Optional[Tensor]       = None  # (B, 4, 4)
    rgb:              Optional[Tensor]       = None  # (B, H, W, 3)
    depth:            Optional[Tensor]       = None  # (B, H, W)
    depth_mask:       Optional[Tensor]       = None  # (B, H, W)
    frame_mask:       Optional[Tensor]       = None  # (B, H, W)
    feat:             Optional[Tensor]       = None  # (B, F)
    featmap:          Optional[Tensor]       = None  # (B, H, W, F)
    featmap_lvls:     Optional[List[Tensor]] = None  # L x (B, H_l, W_l, F)
    # frame-object
    cam_bbox2d:       Optional[Tensor]       = None  # (B, 4)
    cam_bbox3d:       Optional[Tensor]       = None  # (B, 8, 3)
    fo_mask:          Optional[Tensor]       = None  # (B, H, W)
    cam_tform4x4_obj: Optional[Tensor]       = None  # (B, 4, 4)
    # object
    pts3d:              Optional[Tensor]       = None  # (B, N, 3)
    pts3d_feats:        Optional[Tensor]       = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:   Optional[Tensor]       = None  # (B, N) or (B, N, V) bool
    verts3d_feats:      Optional[Tensor]       = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask: Optional[Tensor]       = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj:      Optional[Tensor]       = None  # (B, 4, 4)
    obj_kpts3d:         Optional[Tensor]       = None  # (B, K, 3)
    obj_kpts3d_mask:    Optional[Tensor]       = None  # (B, K)    bool
    category:           Optional[Tensor]       = None  # (B,)  int64
    mesh:               Optional[Mesh]         = None  # shared mesh for all B viewpoints


@dataclass
class SceneObjectBatchModalities:
    """Stacked across B samples. Each sample = 1 scene (T frames) + 1 object."""
    # scene
    cams_intr4x4:      Optional[Tensor]       = None  # (B, T, 4, 4)
    rgbs:              Optional[Tensor]       = None  # (B, T, H, W, 3)
    depths:            Optional[Tensor]       = None  # (B, T, H, W)
    depths_masks:      Optional[Tensor]       = None  # (B, T, H, W)
    scene_masks:       Optional[Tensor]       = None  # (B, T, H, W)
    feats:             Optional[Tensor]       = None  # (B, T, F)
    featmaps:          Optional[Tensor]       = None  # (B, T, H, W, F)
    featmaps_lvls:     Optional[List[Tensor]] = None  # L x (B, T, H_l, W_l, F)
    # scene-object
    cams_bbox2d:       Optional[Tensor]       = None  # (B, T, 4)
    cams_bbox3d:       Optional[Tensor]       = None  # (B, T, 8, 3)
    so_masks:          Optional[Tensor]       = None  # (B, T, H, W)
    cams_tform4x4_obj: Optional[Tensor]       = None  # (B, T, 4, 4)
    # object (time-invariant)
    pts3d:               Optional[Tensor]       = None  # (B, N, 3)
    pts3d_feats:         Optional[Tensor]       = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:    Optional[Tensor]       = None  # (B, N) or (B, N, V) bool
    verts3d_feats:       Optional[Tensor]       = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:  Optional[Tensor]       = None  # (B, N) or (B, N, V) bool
    obj_ncds0c_tform4x4_obj:       Optional[Tensor]       = None  # (B, 4, 4)
    obj_kpts3d:          Optional[Tensor]       = None  # (B, K, 3)
    obj_kpts3d_mask:     Optional[Tensor]       = None  # (B, K)    bool
    category:            Optional[Tensor]       = None  # (B,)  int64


# ── Generic stacking ──────────────────────────────────────────────────────────

def _stack_field(values: list) -> Optional[Tensor]:
    """Stack a list of tensors if all present, else None."""
    if any(v is None for v in values):
        return None
    return torch.stack(values, dim=0)


def collate_frame_objects(
    samples: list[FrameObject],
    include: Optional[set[str]] = None,
) -> FrameObjectBatchModalities:
    """Collate B FrameObject samples into a flat stacked batch."""
    def _get(attr, src):
        vals = [getattr(s, src).__dict__[attr] for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr, src):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, src).__dict__[attr] for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return FrameObjectBatchModalities(
        # frame modalities
        cam_intr4x4      = _get("cam_intr4x4",   "frame_modalities"),
        rgb              = _get("rgb",            "frame_modalities"),
        depth            = _get("depth",          "frame_modalities"),
        depth_mask       = _get("depth_mask",     "frame_modalities"),
        frame_mask       = _get("mask",           "frame_modalities"),
        feat             = _get("feat",           "frame_modalities"),
        featmap          = _get("featmap",        "frame_modalities"),
        featmap_lvls     = _get_lvls("featmap_lvls", "frame_modalities"),
        # frame-object modalities
        cam_bbox2d       = _get("cam_bbox2d",        "frame_object_modalities"),
        cam_bbox3d       = _get("cam_bbox3d",        "frame_object_modalities"),
        fo_mask          = _get("mask",             "frame_object_modalities"),
        cam_tform4x4_obj = _get("cam_tform4x4_obj", "frame_object_modalities"),
        # object modalities
        pts3d               = _get("pts3d",              "object_modalities"),
        pts3d_feats         = _get("pts3d_feats",        "object_modalities"),
        pts3d_feats_mask    = _get("pts3d_feats_mask",   "object_modalities"),
        verts3d_feats       = _get("verts3d_feats",      "object_modalities"),
        verts3d_feats_mask  = _get("verts3d_feats_mask", "object_modalities"),
        obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj", "object_modalities"),
        obj_kpts3d              = _get("obj_kpts3d",               "object_modalities"),
        obj_kpts3d_mask         = _get("obj_kpts3d_mask",          "object_modalities"),
        category                = _stack_field([
            torch.tensor(s.object_modalities.category)
            if s.object_modalities.category is not None else None
            for s in samples
        ]) if (include is None or "category" in include) else None,
    )


def collate_scene_objects(
    samples: list[SceneObject],
    include: Optional[set[str]] = None,
) -> SceneObjectBatchModalities:
    """Collate B SceneObject samples into a flat stacked batch."""
    def _get(attr, src):
        vals = [getattr(s, src).__dict__[attr] for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr, src):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, src).__dict__[attr] for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return SceneObjectBatchModalities(
        # scene modalities
        cams_intr4x4      = _get("cams_intr4x4",  "scene_modalities"),
        rgbs              = _get("rgbs",           "scene_modalities"),
        depths            = _get("depths",         "scene_modalities"),
        depths_masks      = _get("depths_masks",   "scene_modalities"),
        scene_masks       = _get("masks",          "scene_modalities"),
        feats             = _get("feats",          "scene_modalities"),
        featmaps          = _get("featmaps",       "scene_modalities"),
        featmaps_lvls     = _get_lvls("featmaps_lvls", "scene_modalities"),
        # scene-object modalities
        cams_bbox2d       = _get("cams_bbox2d",        "scene_object_modalities"),
        cams_bbox3d       = _get("cams_bbox3d",        "scene_object_modalities"),
        so_masks          = _get("masks",             "scene_object_modalities"),
        cams_tform4x4_obj = _get("cams_tform4x4_obj", "scene_object_modalities"),
        # object modalities
        pts3d               = _get("pts3d",              "object_modalities"),
        pts3d_feats         = _get("pts3d_feats",        "object_modalities"),
        pts3d_feats_mask    = _get("pts3d_feats_mask",   "object_modalities"),
        verts3d_feats       = _get("verts3d_feats",      "object_modalities"),
        verts3d_feats_mask  = _get("verts3d_feats_mask", "object_modalities"),
        obj_ncds0c_tform4x4_obj       = _get("obj_ncds0c_tform4x4_obj",      "object_modalities"),
        obj_kpts3d          = _get("obj_kpts3d",         "object_modalities"),
        obj_kpts3d_mask     = _get("obj_kpts3d_mask",    "object_modalities"),
        category            = _stack_field([
            torch.tensor(s.object_modalities.category)
            if s.object_modalities.category is not None else None
            for s in samples
        ]) if (include is None or "category" in include) else None,
    )


# from torch.utils.data import Dataset, DataLoader

# class PoseEstimationDataset(Dataset):
#     def __getitem__(self, idx) -> FrameObject:
#         # load frame + object for this idx, populate modalities
#         return FrameObject(
#             frame_object_id="fo_0",
#             frame_id="f_0",
#             object_id="obj_0",
#             frame_modalities=FrameModalities(
#                 rgb=torch.rand(480, 640, 3),
#                 cam_intr4x4=torch.eye(4),
#             ),
#             frame_object_modalities=FrameObjectModalities(
#                 cam_bbox2d=torch.tensor([100., 80., 300., 400.]),
#                 cam_tform4x4_obj=torch.eye(4),
#             ),
#             object_modalities=ObjectModalities(
#                 pts3d=torch.rand(1024, 3),
#                 category=7,
#             ),
#         )

# loader = DataLoader(
#     PoseEstimationDataset(),
#     batch_size=8,
#     collate_fn=lambda b: collate_frame_objects(
#         b, include={"rgb", "cam_intr4x4", "bbox2d", "cam_tform4x4_obj", "pts3d"}
#     ),
# )

# for batch in loader:
#     batch.rgb               # (8, 480, 640, 3)
#     batch.cam_tform4x4_obj  # (8, 4, 4)
#     batch.pts3d             # (8, 1024, 3)