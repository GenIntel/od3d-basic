from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from od3d_basic.data.datatypes.mesh import Mesh
from od3d_basic.data.datatypes.frame import _stack_field


def _render_mesh_trimesh(mesh: "Mesh", n_views: int = 6, H: int = 256, W: int = 256) -> "Tensor":
    """Render mesh from n_views azimuth angles using trimesh's offscreen backend."""
    import math
    import numpy as np
    from od3d_basic.io import _mesh_to_trimesh
    import trimesh as tm

    mesh_tm = _mesh_to_trimesh(mesh)
    bounds  = mesh_tm.bounding_box.bounds
    center  = (bounds[0] + bounds[1]) / 2.0
    radius  = float(np.linalg.norm(bounds[1] - bounds[0])) * 0.75

    imgs = []
    for i in range(n_views):
        angle = i * 2.0 * math.pi / n_views
        eye   = center + radius * np.array([math.sin(angle), 0.3, math.cos(angle)])
        # look-at in OpenGL convention (camera looks along -Z)
        fwd   = center - eye;        fwd   /= np.linalg.norm(fwd)   + 1e-8
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right) + 1e-8
        up    = np.cross(right, fwd)
        T = np.eye(4)
        T[:3, 0] = right;  T[:3, 1] = up;  T[:3, 2] = -fwd;  T[:3, 3] = eye

        scene = tm.Scene(mesh_tm)
        scene.camera_transform = T
        try:
            data = scene.save_image(resolution=[W, H])
            img  = torch.from_numpy(
                np.frombuffer(data, dtype=np.uint8).copy().reshape(H, W, 4)[..., :3]
            ).float().permute(2, 0, 1) / 255.0
        except Exception:
            img = torch.zeros(3, H, W)
        imgs.append(img)
    return torch.stack(imgs)  # (N, 3, H, W)


@dataclass(kw_only=True)
class Object:
    object_id:               str
    pts3d:                   Optional[Tensor] = None  # (N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (N, F) or (N, V, F) multi-view
    pts3d_feats_mask:        Optional[Tensor] = None  # (N,) or (N, V) bool
    verts3d_feats:           Optional[Tensor] = None  # (N, F) or (N, V, F) multi-view
    verts3d_feats_mask:      Optional[Tensor] = None  # (N,) or (N, V) bool
    mesh:                    Optional[Mesh]   = None
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (K,)  bool
    category:                Optional[int]    = None
    attributes:              Optional[dict]   = None

    def viz(
        self,
        renderer: str = "pyrender",
        n_views: int = 6,
        H: int = 256,
        W: int = 256,
        server=None,
    ) -> "Optional[Tensor | list]":
        """Render or display the object.

        server=None — renders n_views with the chosen renderer and returns a
                      (3, H, W_total) strip tensor in [0, 1].
        server      — adds mesh, point cloud, and keypoints to the given viser
                      server with checkbox-toggled visibility; returns None.
                      Pass a freshly created ViserServer to display interactively,
                      or a shared server when used inside visualize_dataset.
        """
        if self.mesh is None and self.pts3d is None and self.obj_kpts3d is None:
            return None

        # ── static render (server=None) ───────────────────────────────────────
        if server is None:
            if self.mesh is None:
                return None
            if renderer == "trimesh":
                imgs = _render_mesh_trimesh(self.mesh, n_views=n_views, H=H, W=W)
            else:
                from od3d_basic.cv.visual.viz import sample_uniform_viewpoints, render_mesh_from_viewpoints
                batch = sample_uniform_viewpoints(n_views, mesh=self.mesh)
                imgs  = render_mesh_from_viewpoints(batch, H=H, W=W, renderer=renderer)
            return torch.cat(list(imgs.clamp(0, 1)), dim=2)  # (3, H, W_total)

        # ── populate a viser server ───────────────────────────────────────────
        import numpy as np
        from od3d_basic.io import _mesh_to_trimesh

        mesh_handle = None
        if self.mesh is not None:
            mesh_handle = server.scene.add_mesh_trimesh(
                "/object/mesh", _mesh_to_trimesh(self.mesh)
            )

        pts_handle = None
        if self.pts3d is not None:
            pts    = self.pts3d.cpu().numpy()
            colors = np.full_like(pts, 0.7)
            pts_handle = server.scene.add_point_cloud(
                "/object/pts3d", points=pts, colors=colors, point_size=0.005
            )

        kpts_handle = None
        if self.obj_kpts3d is not None:
            from od3d_basic.cv.visual.viz import _make_kpts_spheres
            mask_np = (self.obj_kpts3d_mask.bool()
                       if self.obj_kpts3d_mask is not None
                       else torch.ones(len(self.obj_kpts3d), dtype=torch.bool)).cpu().numpy()
            kpts_mesh = _make_kpts_spheres(self.obj_kpts3d.cpu().numpy(), mask_np)
            if kpts_mesh is not None:
                kpts_handle = server.scene.add_mesh_trimesh("/object/kpts3d", kpts_mesh)

        gui_folder = server.gui.add_folder("Modalities")
        with gui_folder:
            cb_mesh = server.gui.add_checkbox("Mesh",      initial_value=mesh_handle  is not None)
            cb_pts  = server.gui.add_checkbox("Points",    initial_value=pts_handle   is not None)
            cb_kpts = server.gui.add_checkbox("Keypoints", initial_value=kpts_handle  is not None)

        @cb_mesh.on_update
        def _(_):
            if mesh_handle  is not None: mesh_handle.visible  = cb_mesh.value

        @cb_pts.on_update
        def _(_):
            if pts_handle   is not None: pts_handle.visible   = cb_pts.value

        @cb_kpts.on_update
        def _(_):
            if kpts_handle  is not None: kpts_handle.visible  = cb_kpts.value

        return [h for h in (mesh_handle, pts_handle, kpts_handle, gui_folder) if h is not None]


@dataclass(kw_only=True)
class ObjectPair:
    src_object_id:  str
    trgt_object_id: str
    src_object:     Object
    trgt_object:    Object


@dataclass
class ObjectBatch:
    """Stacked across B Object samples."""
    pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    mesh:                    Optional[Mesh]   = None  # shared mesh for all B samples
    obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    category:                Optional[Tensor] = None  # (B,)      int64


@dataclass
class ObjectPairBatch:
    """Stacked across B ObjectPair samples."""
    src_pts3d:                    Optional[Tensor] = None  # (B, N, 3)
    src_pts3d_feats:              Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    src_pts3d_feats_mask:         Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    src_verts3d_feats:            Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    src_verts3d_feats_mask:       Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    src_mesh:                     Optional[Mesh]   = None  # shared mesh for all B src samples
    src_obj_ncds0c_tform4x4_obj:  Optional[Tensor] = None  # (B, 4, 4)
    src_obj_kpts3d:               Optional[Tensor] = None  # (B, K, 3)
    src_obj_kpts3d_mask:          Optional[Tensor] = None  # (B, K)    bool
    src_category:                 Optional[Tensor] = None  # (B,)      int64
    trgt_pts3d:                   Optional[Tensor] = None  # (B, N, 3)
    trgt_pts3d_feats:             Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    trgt_pts3d_feats_mask:        Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    trgt_verts3d_feats:           Optional[Tensor] = None  # (B, N, F) or (B, N, V, F)
    trgt_verts3d_feats_mask:      Optional[Tensor] = None  # (B, N) or (B, N, V)  bool
    trgt_mesh:                    Optional[Mesh]   = None  # shared mesh for all B trgt samples
    trgt_obj_ncds0c_tform4x4_obj: Optional[Tensor] = None  # (B, 4, 4)
    trgt_obj_kpts3d:              Optional[Tensor] = None  # (B, K, 3)
    trgt_obj_kpts3d_mask:         Optional[Tensor] = None  # (B, K)    bool
    trgt_category:                Optional[Tensor] = None  # (B,)      int64


def collate_object_pairs(
    samples: list[ObjectPair],
    include: Optional[set[str]] = None,
) -> ObjectPairBatch:
    def _get(attr, side: str):
        vals = [getattr(getattr(s, f"{side}_object"), attr) for s in samples]
        if include and f"{side}_{attr}" not in include:
            return None
        return _stack_field(vals)

    def _cat(side: str):
        vals = [getattr(s, f"{side}_object").category for s in samples]
        key = f"{side}_category"
        if include and key not in include:
            return None
        return _stack_field([
            torch.tensor(v) if v is not None else None for v in vals
        ])

    return ObjectPairBatch(
        src_pts3d                   = _get("pts3d",                   "src"),
        src_pts3d_feats             = _get("pts3d_feats",             "src"),
        src_pts3d_feats_mask        = _get("pts3d_feats_mask",        "src"),
        src_verts3d_feats           = _get("verts3d_feats",           "src"),
        src_verts3d_feats_mask      = _get("verts3d_feats_mask",      "src"),
        src_obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj","src"),
        src_obj_kpts3d              = _get("obj_kpts3d",              "src"),
        src_obj_kpts3d_mask         = _get("obj_kpts3d_mask",         "src"),
        src_category                = _cat("src"),
        trgt_pts3d                   = _get("pts3d",                   "trgt"),
        trgt_pts3d_feats             = _get("pts3d_feats",             "trgt"),
        trgt_pts3d_feats_mask        = _get("pts3d_feats_mask",        "trgt"),
        trgt_verts3d_feats           = _get("verts3d_feats",           "trgt"),
        trgt_verts3d_feats_mask      = _get("verts3d_feats_mask",      "trgt"),
        trgt_obj_ncds0c_tform4x4_obj = _get("obj_ncds0c_tform4x4_obj","trgt"),
        trgt_obj_kpts3d              = _get("obj_kpts3d",              "trgt"),
        trgt_obj_kpts3d_mask         = _get("obj_kpts3d_mask",         "trgt"),
        trgt_category                = _cat("trgt"),
    )


def collate_objects(
    samples: list[Object],
    include: Optional[set[str]] = None,
) -> ObjectBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    return ObjectBatch(
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
