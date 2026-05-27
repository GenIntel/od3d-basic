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
    render_frames: int = 6,
    renderer: str = "pyrender",
) -> None:
    """Browse a dataset interactively with Prev / Next navigation.

    render=False (default): passes the viser server to item.viz(server=server)
    so items add their 3D content (mesh, point cloud, keypoints) directly.
    render=True: renders render_frames viewpoints with the chosen renderer and
    shows the resulting strip as the viser background image instead.
    """
    try:
        import viser
        import numpy as np
    except ImportError:
        print("Install viser: pip install viser")
        return

    server = viser.ViserServer()
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
        if render and render_frames > 0:
            import torch
            mesh = getattr(item, "mesh", None)
            if mesh is not None:
                batch = sample_uniform_viewpoints(render_frames, mesh=mesh)
                imgs = render_mesh_from_viewpoints(batch, renderer=renderer)      # (N, 3, H, W)
                img  = torch.cat(list(imgs.clamp(0, 1)), dim=2)                   # (3, H, W_total)
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
                server.scene.set_background_image(img_np, format="png")
        else:
            result = item.viz(server=server)
            if isinstance(result, list):
                handles.extend(result)
        label.value = f"[{i + 1}/{n}]  {item_id}"
        print(f"  [{i + 1}/{n}] {item_id}")

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
    dist: float = 2.5,
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
) -> "Tensor":
    """Render mesh from B viewpoints using pyrender. Returns (B, 3, H, W) in [0, 1]."""
    import torch
    from od3d_basic.io import _mesh_to_trimesh
    from od3d_basic.cv.visual.show import render_trimesh_to_tensor
    mesh_tm = _mesh_to_trimesh(mesh)
    B = cam_tform4x4_obj.shape[0]
    frames = []
    for b in range(B):
        rgb, _ = render_trimesh_to_tensor(mesh_tm, cam_intr4x4[b], cam_tform4x4_obj[b], H=H, W=W)
        frames.append(rgb)
    return torch.stack(frames, dim=0)  # (B, 3, H, W)


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

    znear, zfar = 0.01, 10000.0

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
        tex = mesh.texture.permute(1, 2, 0).to(device=device, dtype=torch.float32)
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
    return color.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)


def render_mesh_from_viewpoints(
    batch: "FrameObjectBatch",
    H: int = 512,
    W: int = 512,
    renderer: str = "pyrender",
) -> "Tensor":
    """
    Render batch.mesh from viewpoints in batch.cam_tform4x4_obj.

    Args:
        batch: FrameObjectBatch with mesh (shared) and cam_tform4x4_obj (B, 4, 4).
               Optionally cam_intr4x4 (B, 4, 4) or (4, 4); defaults to 25° FOV.
        H, W:  Output image size.
        renderer: "pyrender" or "nvdiffrast".

    Returns:
        (B, 3, H, W) float32 RGB in [0, 1].
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
