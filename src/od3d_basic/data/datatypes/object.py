from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from od3d_basic.data.datatypes.mesh import Mesh
from od3d_basic.data.datatypes.frame import _stack_field


def _draw_kpts2d_on_imgs(
    imgs: "Tensor",           # (B, 3, H, W) float32 [0,1]
    kpts2d: "Tensor",         # (B, K, 2) float32  [u, v] pixel coords
    mask: "Optional[Tensor]", # (K,) bool or None
    radius: int = 5,
) -> "Tensor":
    """Draw HSV-coloured filled circles at keypoint locations in-place (cloned copy)."""
    import colorsys
    B, _, H, W = imgs.shape
    K = kpts2d.shape[1]
    result = imgs.clone()
    ys = torch.arange(H, dtype=torch.float32, device=imgs.device)
    xs = torch.arange(W, dtype=torch.float32, device=imgs.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    for k in range(K):
        if mask is not None and not mask[k]:
            continue
        r, g, b = colorsys.hsv_to_rgb(k / max(K, 1), 0.9, 0.88)
        color = torch.tensor([r, g, b], dtype=torch.float32, device=imgs.device)
        for b_i in range(B):
            u, v = kpts2d[b_i, k, 0], kpts2d[b_i, k, 1]
            circle = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2  # (H, W)
            result[b_i, :, circle] = color[:, None]
    return result


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
    category_id:             Optional[int]    = None
    attributes:              Optional[dict]   = None

    def render_modalities(
        self,
        renderer: str = "pyrender",
        n_views: int = 4,
        H: int = 256,
        W: int = 256,
    ) -> "Optional[dict]":
        """Return dict of rendered modalities {'rgb','depth','normals'}, each (N,3,H,W)."""
        if self.mesh is None:
            return None
        from od3d_basic.data.viz import sample_uniform_viewpoints, render_mesh_from_viewpoints
        from od3d_basic.cv.visual.show import get_default_camera_intrinsics_from_img_size

        batch = sample_uniform_viewpoints(n_views, mesh=self.mesh)
        modalities = render_mesh_from_viewpoints(batch, H=H, W=W, renderer=renderer)

        if self.obj_kpts3d is not None:
            cam_tform4x4_obj = batch.cam_tform4x4_obj  # (B, 4, 4)
            B = cam_tform4x4_obj.shape[0]
            cam_intr4x4 = batch.cam_intr4x4
            if cam_intr4x4 is None:
                cam_intr4x4 = get_default_camera_intrinsics_from_img_size(W, H).unsqueeze(0).expand(B, -1, -1)
            elif cam_intr4x4.dim() == 2:
                cam_intr4x4 = cam_intr4x4.unsqueeze(0).expand(B, -1, -1)

            kpts3d = self.obj_kpts3d.float()       # (K, 3)
            K = kpts3d.shape[0]
            kpts3d_h = torch.cat([kpts3d, torch.ones(K, 1)], dim=1)  # (K, 4)
            kpts_cam = (cam_tform4x4_obj @ kpts3d_h.T).permute(0, 2, 1)  # (B, K, 4)
            kpts_xyz = kpts_cam[..., :3]                                   # (B, K, 3)
            kpts_proj = torch.bmm(cam_intr4x4[:, :3, :3],
                                  kpts_xyz.permute(0, 2, 1)).permute(0, 2, 1)  # (B, K, 3)
            z = kpts_proj[..., 2:3].clamp(min=1e-6)
            kpts2d = kpts_proj[..., :2] / z  # (B, K, 2)  [u, v]

            modalities["rgb"] = _draw_kpts2d_on_imgs(
                modalities["rgb"], kpts2d,
                mask=self.obj_kpts3d_mask,
                radius=max(H, W) // 50,
            )

        return modalities

    def viz(
        self,
        renderer: str = "pyrender",
        n_views: int = 6,
        H: int = 256,
        W: int = 256,
        server=None,
        node_prefix: str = "/object",
        gui_label: str = "Modalities",
        position_offset: tuple = (0.0, 0.0, 0.0),
    ) -> "Optional[Tensor | list]":
        """Render or display the object.

        server=None — renders n_views with the chosen renderer and returns a
                      (3, H, W_total) strip tensor in [0, 1].
        server      — adds mesh, point cloud, and keypoints to the given viser
                      server; returns the list of handles for later removal.
                      node_prefix / gui_label / position_offset allow ObjectPair
                      to place src and trgt side-by-side in the same scene.
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
                imgs = self.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)["rgb"]
            return torch.cat(list(imgs.clamp(0, 1)), dim=2)  # (3, H, W_total)

        # ── populate a viser server ───────────────────────────────────────────
        import numpy as np
        from od3d_basic.io import _mesh_to_trimesh

        mesh_handle = None
        if self.mesh is not None:
            mesh_handle = server.scene.add_mesh_trimesh(
                f"{node_prefix}/mesh", _mesh_to_trimesh(self.mesh),
                position=position_offset,
            )

        pts_handle = None
        if self.pts3d is not None:
            pts    = self.pts3d.cpu().numpy()
            colors = np.full_like(pts, 0.7)
            pts_handle = server.scene.add_point_cloud(
                f"{node_prefix}/pts3d", points=pts, colors=colors, point_size=0.005,
                position=position_offset,
            )

        kpts_handle = None
        if self.obj_kpts3d is not None:
            from od3d_basic.data.viz import _make_kpts_spheres
            mask_np = (self.obj_kpts3d_mask.bool()
                       if self.obj_kpts3d_mask is not None
                       else torch.ones(len(self.obj_kpts3d), dtype=torch.bool)).cpu().numpy()
            kpts_mesh = _make_kpts_spheres(self.obj_kpts3d.cpu().numpy(), mask_np)
            if kpts_mesh is not None:
                kpts_handle = server.scene.add_mesh_trimesh(
                    f"{node_prefix}/kpts3d", kpts_mesh,
                    position=position_offset,
                )

        gui_folder = server.gui.add_folder(gui_label)
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

    def render_modalities(
        self,
        renderer: str = "pyrender",
        n_views: int = 4,
        H: int = 256,
        W: int = 256,
    ) -> "Optional[dict]":
        """Return dict of modalities with src and trgt concatenated side-by-side (dim=3)."""
        import torch
        src_mods  = self.src_object.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)
        trgt_mods = self.trgt_object.render_modalities(renderer=renderer, n_views=n_views, H=H, W=W)
        if src_mods is None and trgt_mods is None:
            return None
        base = src_mods or trgt_mods
        return {
            key: torch.cat(
                [m[key] for m in (src_mods, trgt_mods) if m is not None and key in m],
                dim=2,  # concatenate along height → (N, 3, H_src + H_trgt, W)
            )
            for key in base
        }

    def viz(
        self,
        renderer: str = "pyrender",
        n_views: int = 6,
        H: int = 256,
        W: int = 256,
        server=None,
        gap: float = 0.5,
    ) -> "Optional[Tensor | list]":
        """Render or display src and trgt objects side-by-side.

        server=None — renders both objects and concatenates their strips
                      width-wise, returning (3, H, W_src + W_trgt).
        server      — adds src at origin and trgt offset along +x by the
                      src mesh x-extent plus gap; returns all handles.
        """
        if server is None:
            src_img  = self.src_object.viz(renderer=renderer, n_views=n_views, H=H, W=W)
            trgt_img = self.trgt_object.viz(renderer=renderer, n_views=n_views, H=H, W=W)
            imgs = [i for i in (src_img, trgt_img) if i is not None]
            return torch.cat(imgs, dim=2) if imgs else None  # (3, H, W_total)

        # compute x offset so trgt sits next to src
        src_mesh = self.src_object.mesh
        if src_mesh is not None:
            x = src_mesh.verts[:, 0]
            x_offset = (x.max() - x.min()).item() + gap
        else:
            x_offset = 2.0 + gap

        src_handles  = self.src_object.viz(
            server=server, node_prefix="/src",  gui_label="Source",
            position_offset=(0.0, 0.0, 0.0),
        ) or []
        trgt_handles = self.trgt_object.viz(
            server=server, node_prefix="/trgt", gui_label="Target",
            position_offset=(x_offset, 0.0, 0.0),
        ) or []
        return src_handles + trgt_handles


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
