from __future__ import annotations
import colorsys
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import Tensor
    from od3d_basic.data.datatypes import Mesh, FrameObjectBatch


def _make_kpts_spheres(kpts_np, mask_np, radius: float = 0.02):
    try:
        import trimesh
        import numpy as np
    except ImportError:
        return None

    template = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    n_total = len(kpts_np)
    meshes = []
    for i in range(n_total):
        if not mask_np[i]:
            continue
        r, g, b = colorsys.hsv_to_rgb(i / max(n_total, 1), 0.9, 0.88)
        color = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        s = template.copy()
        s.apply_translation(kpts_np[i])
        s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1))
        meshes.append(s)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def visualize_dataset(
    dataset,
    render: bool = False,
    render_frames: int = 4,
    renderer: str = "pyrender",
) -> None:
    """Browse a dataset interactively with Prev / Next navigation.

    render=False (default): passes the viser server to item.viz(server=server)
    so items add their 3D content (mesh, point cloud, keypoints) directly.
    render=True: renders render_frames viewpoints with the chosen renderer and
    displays the result via show_imgs (no viser required).
    """
    if render:
        import torch
        import tkinter as _tk
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons

        _root = _tk.Tk(); _root.withdraw()
        _disp_h = _root.winfo_screenheight() // 2
        _disp_w = _root.winfo_screenwidth()  // 2
        _root.destroy()

        n = len(dataset)
        for i in range(n):
            item = dataset[i]
            item_id = getattr(
                item, "object_id",
                getattr(item, "frame_id",
                        getattr(item, "scene_id", str(i))),
            )
            category = getattr(item, "category", None)
            cat_str  = f"  cat={category}" if category is not None else ""
            print(f"  [{i + 1}/{n}] {item_id}{cat_str}")
            render_fn = getattr(item, "render_modalities", None)
            if render_fn is not None:
                modalities = render_fn(renderer=renderer, n_views=render_frames)
            else:
                mesh = getattr(item, "mesh", None)
                if mesh is None:
                    continue
                batch_vp = sample_uniform_viewpoints(render_frames, mesh=mesh)
                modalities = render_mesh_from_viewpoints(batch_vp, renderer=renderer)
            if modalities is None:
                continue
            keys = list(modalities.keys())

            def _to_strip(imgs):
                return torch.cat(list(imgs.clamp(0, 1)), dim=2).permute(1, 2, 0).cpu().numpy()

            fig, ax = plt.subplots(figsize=(_disp_w / 100, _disp_h / 100))
            fig.suptitle(f"{item_id}{cat_str}", fontsize=9)
            plt.subplots_adjust(left=0.15, top=0.93)
            ax.axis("off")
            img_display = ax.imshow(_to_strip(modalities[keys[0]]))

            radio_ax = fig.add_axes([0.01, 0.4, 0.12, len(keys) * 0.07])
            radio = RadioButtons(radio_ax, keys, active=0)

            def _on_select(label, _disp=img_display, _mod=modalities, _fig=fig):
                _disp.set_data(_to_strip(_mod[label]))
                _disp.autoscale()
                _fig.canvas.draw_idle()

            radio.on_clicked(_on_select)
            plt.show()
        return

    # ── viser path ────────────────────────────────────────────────────────────
    try:
        import viser
    except ImportError:
        print("Install viser: pip install viser")
        return

    server = viser.ViserServer()
    server.scene.add_light_ambient("/ambient", intensity=3.0)
    n = len(dataset)
    idx = [0]
    handles: list = []

    def _clear() -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _load(i: int) -> None:
        _clear()
        item = dataset[i]
        item_id = getattr(
            item, "object_id",
            getattr(item, "frame_id",
                    getattr(item, "scene_id", str(i))),
        )
        result = item.viz(server=server)
        if isinstance(result, list):
            handles.extend(result)
        category = getattr(item, "category", None)
        cat_str  = f"  cat={category}" if category is not None else ""
        label.value = f"[{i + 1}/{n}]  {item_id}{cat_str}"
        print(f"  [{i + 1}/{n}] {item_id}{cat_str}")

    with server.gui.add_folder("Navigation"):
        label    = server.gui.add_text("Item",   initial_value="loading…")
        btn_prev = server.gui.add_button("← Prev")
        btn_next = server.gui.add_button("Next →")

    @btn_prev.on_click
    def _(_):
        idx[0] = (idx[0] - 1) % n
        _load(idx[0])

    @btn_next.on_click
    def _(_):
        idx[0] = (idx[0] + 1) % n
        _load(idx[0])

    _load(0)
    print(f"\nViser running at http://localhost:{server.get_port()}")
    print("Use Prev / Next in the panel to browse. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")


def sample_uniform_viewpoints(
    n: int,
    dist: float = 5.0,
    mesh: "Optional[Mesh]" = None,
) -> "FrameObjectBatch":
    """Returns FrameObjectBatch with cam_tform4x4_obj (n, 4, 4) sampled over a sphere."""
    from od3d_basic.cv.geometry.transform import get_cam_tform4x4_obj_for_viewpoints_count
    from od3d_basic.data.datatypes import FrameObjectBatch
    cam_tform4x4_obj = get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=n, dist=dist)
    return FrameObjectBatch(cam_tform4x4_obj=cam_tform4x4_obj, mesh=mesh)


def _render_mesh_pyrender(
    mesh: "Mesh",
    cam_tform4x4_obj: "Tensor",  # (B, 4, 4)
    cam_intr4x4: "Tensor",       # (B, 4, 4)
    H: int,
    W: int,
) -> "dict[str, Tensor]":
    """Render mesh from B viewpoints using pyrender. Returns dict of (B, 3, H, W) in [0, 1]."""
    import torch
    import numpy as np
    import trimesh as _tm
    from od3d_basic.io import _mesh_to_trimesh
    from od3d_basic.cv.visual.show import render_trimesh_to_tensor

    mesh_tm = _mesh_to_trimesh(mesh)

    # normals mesh: vertex colors encode world-space normals mapped to [0, 1]
    vn = mesh_tm.vertex_normals.astype(np.float32)
    vc_rgba = np.concatenate(
        [((vn * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8),
         np.full((len(vn), 1), 255, np.uint8)], axis=1
    )
    normals_tm = _tm.Trimesh(vertices=mesh_tm.vertices, faces=mesh_tm.faces,
                              vertex_colors=vc_rgba, process=False)

    B = cam_tform4x4_obj.shape[0]
    rgb_frames, depth_frames, normal_frames = [], [], []
    for b in range(B):
        intr, tform = cam_intr4x4[b], cam_tform4x4_obj[b]
        rgb, depth = render_trimesh_to_tensor(mesh_tm, intr, tform, H=H, W=W)
        normal_rgb, _ = render_trimesh_to_tensor(normals_tm, intr, tform, H=H, W=W)
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        normal_frames.append(normal_rgb)

    rgb    = torch.stack(rgb_frames,    dim=0)  # (B, 3, H, W)
    depth  = torch.stack(depth_frames,  dim=0)  # (B, 1, H, W)
    normals = torch.stack(normal_frames, dim=0)  # (B, 3, H, W)

    d = depth[:, 0]
    d_max = d[d > 0].max() if (d > 0).any() else torch.tensor(1.0)
    depth_rgb = (d / d_max).clamp(0, 1).unsqueeze(1).expand(-1, 3, -1, -1)

    return {"rgb": rgb, "depth": depth_rgb, "normals": normals}


def _render_mesh_nvdiffrast(
    mesh: "Mesh",
    cam_tform4x4_obj: "Tensor",  # (B, 4, 4)
    cam_intr4x4: "Tensor",       # (B, 4, 4)
    H: int,
    W: int,
) -> "Tensor":
    """Render mesh from B viewpoints using nvdiffrast. Returns (B, 3, H, W) in [0, 1]."""
    import os
    import torch
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import nvdiffrast.torch as dr
    from od3d_basic.cv.visual.show import OPEN3D_CAM_TFORM_CAM

    device = torch.device("cuda")
    cam_tform4x4_obj = cam_tform4x4_obj.to(device)
    cam_intr4x4 = cam_intr4x4.to(device)
    B = cam_tform4x4_obj.shape[0]

    znear, zfar = 0.001, 10000.0

    cams_persp4x4 = cam_intr4x4.clone().to(device=device, dtype=torch.float32)
    cams_persp4x4[:, 0, 2] = -cams_persp4x4[:, 0, 2]
    cams_persp4x4[:, 1, 2] = -cams_persp4x4[:, 1, 2]
    cams_persp4x4[:, 2, 2] = 0.0
    cams_persp4x4[:, 2, 3] = 1.0
    cams_persp4x4[:, 3, 2] = 1.0
    cams_persp4x4[:, 3, 3] = 0.0
    cams_persp4x4[:, 1, 2] = -(H + cams_persp4x4[:, 1, 2])

    top, bottom, left, right = 0, max(H, 1), 0, max(W, 1)
    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    U = -2.0 * znear * zfar / (zfar - znear)
    V_coef = -(zfar + znear) / (zfar - znear)
    ndc_mat = torch.tensor([
        [2.0 / (right - left), 0.0, 0.0, -tx],
        [0.0, 2.0 / (top - bottom), 0.0, -ty],
        [0.0, 0.0, U, V_coef],
        [0.0, 0.0, 0.0, -1.0],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 4)

    open3d_tform = OPEN3D_CAM_TFORM_CAM.clone().unsqueeze(0).to(device=device, dtype=torch.float32)
    cams_proj4x4 = (ndc_mat @ cams_persp4x4) @ (open3d_tform @ cam_tform4x4_obj)  # (B, 4, 4)

    verts = mesh.verts.to(device=device, dtype=torch.float32)  # (V, 3)
    V_count = verts.shape[0]
    verts_h = torch.cat([verts, torch.ones(V_count, 1, device=device, dtype=torch.float32)], dim=-1)
    verts_clip = (cams_proj4x4[:, None] @ verts_h[None, :, :, None])[..., 0]  # (B, V, 4)

    faces = mesh.faces.to(device=device, dtype=torch.int32)  # (F, 3)

    glctx = dr.RasterizeCudaContext(device=device)
    rast_out, _ = dr.rasterize(glctx, verts_clip.contiguous(), faces.contiguous(), resolution=[H, W])

    if mesh.texture is not None and mesh.verts_uvs is not None:
        uv = mesh.verts_uvs.to(device=device, dtype=torch.float32).clone()
        uv[:, 1] = 1.0 - uv[:, 1]
        uv_idx = (mesh.faces_uvs if mesh.faces_uvs is not None else faces).to(device=device, dtype=torch.int32)
        texc, _ = dr.interpolate(uv.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, uv_idx.contiguous())
        tex = mesh.texture.flip(1).permute(1, 2, 0).to(device=device, dtype=torch.float32)
        color = dr.texture(tex.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), texc.contiguous(), filter_mode="linear")
    elif mesh.vert_colors is not None:
        vc = mesh.vert_colors.to(device=device, dtype=torch.float32)
        color, _ = dr.interpolate(vc.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces.contiguous())
    else:
        gray = torch.full((V_count, 3), 0.7, device=device, dtype=torch.float32)
        color, _ = dr.interpolate(gray.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces.contiguous())

    try:
        color = dr.antialias(color.contiguous(), rast_out, verts_clip.contiguous(), faces.contiguous())
    except Exception:
        pass
    mask = (rast_out[..., 3:4] > 0).float()
    color = color * mask
    rgb = color.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

    # ── Depth ─────────────────────────────────────────────────────────────────
    verts_cam = (cam_tform4x4_obj[:, None] @ verts_h[None, :, :, None])[..., 0]  # (B, V, 4)
    verts_z = verts_cam[:, :, 2:3]  # positive depth in Open3D convention (B, V, 1)
    depth_interp, _ = dr.interpolate(verts_z.contiguous(), rast_out, faces)  # (B, H, W, 1)
    depth_masked = depth_interp * mask
    d_max = depth_masked.max()
    depth_rgb = (depth_masked / (d_max + 1e-6)).expand(-1, -1, -1, 3).permute(0, 3, 1, 2).contiguous()

    # ── Normals ───────────────────────────────────────────────────────────────
    import numpy as np
    verts_np = mesh.verts.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    v0, v1, v2 = verts_np[faces_np[:, 0]], verts_np[faces_np[:, 1]], verts_np[faces_np[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    vn = np.zeros_like(verts_np)
    for i in range(3):
        np.add.at(vn, faces_np[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8
    vn_t = torch.from_numpy(vn.astype(np.float32)).to(device)  # (V, 3)
    normals_interp, _ = dr.interpolate(
        vn_t.unsqueeze(0).expand(B, -1, -1).contiguous(), rast_out, faces
    )  # (B, H, W, 3)
    normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1)
    normals_rgb = ((normals_interp * 0.5 + 0.5) * mask).permute(0, 3, 1, 2).contiguous()

    return {"rgb": rgb, "depth": depth_rgb, "normals": normals_rgb}


def render_mesh_from_viewpoints(
    batch: "FrameObjectBatch",
    H: int = 512,
    W: int = 512,
    renderer: str = "pyrender",
) -> "dict[str, Tensor]":
    """
    Render batch.mesh from viewpoints in batch.cam_tform4x4_obj.

    Args:
        batch: FrameObjectBatch with mesh (shared) and cam_tform4x4_obj (B, 4, 4).
               Optionally cam_intr4x4 (B, 4, 4) or (4, 4); defaults to 25° FOV.
        H, W:  Output image size.
        renderer: "pyrender" or "nvdiffrast".

    Returns:
        dict with keys "rgb", "depth", "normals", each (B, 3, H, W) float32 in [0, 1].
    """
    import torch
    from od3d_basic.cv.visual.show import get_default_camera_intrinsics_from_img_size

    cam_tform4x4_obj = batch.cam_tform4x4_obj
    B = cam_tform4x4_obj.shape[0]

    cam_intr4x4 = batch.cam_intr4x4
    if cam_intr4x4 is None:
        cam_intr4x4 = get_default_camera_intrinsics_from_img_size(W, H).unsqueeze(0).expand(B, -1, -1)
    elif cam_intr4x4.dim() == 2:
        cam_intr4x4 = cam_intr4x4.unsqueeze(0).expand(B, -1, -1)

    if renderer == "nvdiffrast":
        return _render_mesh_nvdiffrast(batch.mesh, cam_tform4x4_obj, cam_intr4x4, H, W)
    return _render_mesh_pyrender(batch.mesh, cam_tform4x4_obj, cam_intr4x4, H, W)
