import logging

logger = logging.getLogger(__name__)
import torch
from od3d_basic.cv.geometry.grid import get_pxl2d
from od3d_basic.cv.geometry.transform import proj3d2d_broadcast, tform4x4_broadcast


def get_scale_bbox_pts3d_to_image(
    cam_intr4x4,
    cam_tform4x4_obj,
    pts3d,
    img_width,
    img_height,
    pts3d_mask=None,
):
    """
    Args:
        cam_intr4x4 (torch.Tensor): ...x4x4
        cam_tform4x4_obj (torch.Tensor): ...x4x4
        pts3d (torch.Tensor): ...xNx3
        pts3d_mask (torch.Tensor): ...xN
        img_width (int)
        img_height (int)
        cx (float)
        cy (float)
    Returns:
        scale (torch.Tensor): ...x4, [scale_x_min, scale_y_min, scale_x_max, scale_y_max]
    """

    pts3d_bbox = get_bbox_from_mask_pts3d(
        cam_intr4x4,
        cam_tform4x4_obj,
        pts3d,
        pts3d_mask,
    )
    img_bbox = get_bbox_from_width_and_height(
        width=img_width,
        height=img_height,
        device=pts3d.device,
        dtype=pts3d.dtype,
    )
    cx = cam_intr4x4[..., 0, 2]
    cy = cam_intr4x4[..., 1, 2]

    scale = get_scale_bbox_A_relative_to_B(
        bboxA=pts3d_bbox,
        bboxB=img_bbox,
        cx=cx,
        cy=cy,
    )
    return scale


def get_bbox_from_mask_pts3d(cam_intr4x4, cam_tform4x4_obj, pts3d, pts3d_mask=None):
    """
    Args:
        cam_intr4x4 (torch.Tensor): ...x4x4
        cam_tform4x4_obj (torch.Tensor): ...x4x4
        pts3d (torch.Tensor): ...xNx3
        pts3d_mask (torch.Tensor): ...xN
    Returns:
        bbox (torch.Tensor): ...x4, [x0, y0, x1, y1]
    """

    cam_proj4x4 = tform4x4_broadcast(cam_intr4x4, cam_tform4x4_obj)
    pxl2d = proj3d2d_broadcast(proj4x4=cam_proj4x4, pts3d=pts3d)

    bbox = get_bbox_from_pxl2d(pxl2d_mask=pts3d_mask, pxl2d=pxl2d)
    return bbox


def get_bbox_from_width_and_height(width, height, device=None, dtype=None):
    """
    Args:
        width (int): ...
        height (int): ...
    Returns:
        bbox (torch.Tensor): ...x4, [x0, y0, x1, y1]
    """
    x0 = torch.Tensor([0.0])
    x1 = torch.Tensor([width])
    y0 = torch.Tensor([0.0])
    y1 = torch.Tensor([height])
    bbox = torch.stack([x0, y0, x1, y1], dim=-1)
    if device is not None:
        bbox = bbox.to(device=device)
    if dtype is not None:
        bbox = bbox.to(dtype=dtype)
    return bbox


def get_bbox_from_pxl2d(pxl2d_mask=None, pxl2d=None):
    """
    Args:
        pxl2d (torch.Tensor): ...x2
        pxl2d_mask (torch.Tensor): BxCxV, BxCxHxW
    Returns:
        bbox (torch.Tensor): BxCx4, [x0, y0, x1, y1]
    """
    if pxl2d is None:
        device = pxl2d_mask.device
        dtype = pxl2d_mask.dtype
        H, W = pxl2d_mask.shape[-2:]
        pxl2d = get_pxl2d(H=H, W=W, device=device, dtype=dtype)  # HxWx2
        pxl2d = pxl2d[(None,) * (pxl2d_mask.dim() - 2)].expand(
            pxl2d_mask.shape[:-2] + pxl2d.shape[-2:],
        )  # ...xHxWx2
    else:
        pxl2d = pxl2d.clone()

    if pxl2d_mask is None:
        pxl2d_mask = torch.full_like(pxl2d[..., 0], True, dtype=torch.bool)

    # get maximum
    pxl2d[~pxl2d_mask] = float("-inf")
    x_max = pxl2d[..., 0].max(dim=-1).values
    y_max = pxl2d[..., 1].max(dim=-1).values

    # get minimum
    pxl2d[~pxl2d_mask] = float("+inf")
    x_min = pxl2d[..., 0].min(dim=-1).values
    y_min = pxl2d[..., 1].min(dim=-1).values

    bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    return bbox


def get_scale_bbox_A_relative_to_B(bboxA, bboxB, cx, cy, eps=1e-8):
    """
    Args:
        bboxA (torch.Tensor): ...x4, [x0, y0, x1, y1]
        bboxB (torch.Tensor): ...x4, [x0, y0, x1, y1]
        cx (torch.Tensor): ...
        cy (torch.Tensor): ...
    Returns:
        scale (torch.Tensor): ...x4, [scale_x_min, scale_y_min, scale_x_max, scale_y_max]
    """
    device = bboxA.device
    scale_x_min = (
        (bboxA[..., 0] - cx) / (bboxB[..., 0] - cx)
        if (bboxB[..., 0] - cx).abs() > eps
        else torch.full_like(bboxA[..., 0], float("+inf")).to(device=device)
    )
    scale_x_max = (
        (bboxA[..., 2] - cx) / (bboxB[..., 2] - cx)
        if (bboxB[..., 2] - cx).abs() > eps
        else torch.full_like(bboxA[..., 2], float("+inf")).to(device=device)
    )
    scale_y_min = (
        (bboxA[..., 1] - cy) / (bboxB[..., 1] - cy)
        if (bboxB[..., 1] - cy).abs() > eps
        else torch.full_like(bboxA[..., 1], float("+inf")).to(device=device)
    )
    scale_y_max = (
        (bboxA[..., 3] - cy) / (bboxB[..., 3] - cy)
        if (bboxB[..., 3] - cy).abs() > eps
        else torch.full_like(bboxA[..., 3], float("+inf")).to(device=device)
    )

    scale = torch.stack([scale_x_min, scale_y_min, scale_x_max, scale_y_max], dim=-1)
    return scale


def depth_from_mesh_and_box(
    b_cams_multiview_intr4x4,
    b_cams_multiview_tform4x4_obj,
    meshes,
    labels,
    mask,
    downsample_rate,
    multiview=False,
):
    """
    Args:
        b_cams_multiview_intr4x4(torch.Tensor): BxCx4x4/Bx4x4
        b_cams_multiview_tform4x4_obj(torch.Tensor): BxCx4x4
        meshes(Meshes): M Meshes
        labels(torch.LongTensor): B labels in range [0, M-1]
        mask(torch.Tensor): BxCxHxW
    Returns:
        depth(torch.Tensor): BxC
    """

    # fx X / (Z s) + cx = x -> (x_proj - cx) / s = (x_box - cx) -> s = (x_proj - cx) / (x_box - cx)
    # fy Y / (Z s) + cy = y -> (y_proj - cy) / s = (y_box - cy) -> s = (y_proj - cy) / (y_box - cy)
    # Z' = Z s

    device = mask.device
    B, C = b_cams_multiview_tform4x4_obj.shape[:2]
    H = mask.shape[-2] * downsample_rate
    W = mask.shape[-1] * downsample_rate
    imgs_sizes = torch.tensor([H, W], dtype=torch.float32, device=device)
    cx = b_cams_multiview_intr4x4[..., 0, 2][:, None]
    cy = b_cams_multiview_intr4x4[..., 1, 2][:, None]

    mesh_verts2d, mesh_verts2d_vsbl = meshes.verts2d(
        cams_tform4x4_obj=b_cams_multiview_tform4x4_obj,
        cams_intr4x4=b_cams_multiview_intr4x4[:, None],
        mesh_ids=labels,
        imgs_sizes=imgs_sizes,
        down_sample_rate=downsample_rate,
        broadcast_batch_and_cams=True,
    )

    # get largest x and y from vert2d with BxCxVx2 and mask with BxCxV
    mesh_verts2d_masked_mask = mesh_verts2d_vsbl[..., None].repeat(1, 1, 1, 2)
    mesh_verts2d_masked = torch.full_like(mesh_verts2d, float("-inf"))
    mesh_verts2d_masked[mesh_verts2d_masked_mask] = mesh_verts2d[
        mesh_verts2d_masked_mask
    ]
    mesh_verts2d_x_max = (
        mesh_verts2d_masked[..., 0].max(dim=-1).values * downsample_rate
    ).clamp(0, W - 1)
    mesh_verts2d_y_max = (
        mesh_verts2d_masked[..., 1].max(dim=-1).values * downsample_rate
    ).clamp(0, H - 1)
    mesh_verts2d_masked = torch.full_like(mesh_verts2d, float("+inf"))
    mesh_verts2d_masked[mesh_verts2d_masked_mask] = mesh_verts2d[
        mesh_verts2d_masked_mask
    ]
    mesh_verts2d_x_min = (
        mesh_verts2d_masked[..., 0].min(dim=-1).values * downsample_rate
    ).clamp(0, W - 1)
    mesh_verts2d_y_min = (
        mesh_verts2d_masked[..., 1].min(dim=-1).values * downsample_rate
    ).clamp(0, H - 1)

    mask_pxl2d = get_pxl2d(
        H=mask.shape[-2],
        W=mask.shape[-1],
        dtype=float,
        device=device,
        B=B,
    )[:, None]
    mask_pxl2d_masked_mask = mask[..., None].repeat(1, 1, 1, 1, 2)
    mask_pxl2d_masked = torch.full_like(mask_pxl2d, float("-inf"))
    mask_pxl2d_masked[mask_pxl2d_masked_mask] = mask_pxl2d[mask_pxl2d_masked_mask]
    mask_verts2d_x_max = (
        mask_pxl2d_masked[..., 0].flatten(2).max(dim=-1).values * downsample_rate
    ).clamp(0, W - 1)
    mask_verts2d_y_max = (
        mask_pxl2d_masked[..., 1].flatten(2).max(dim=-1).values * downsample_rate
    ).clamp(0, H - 1)
    mask_pxl2d_masked = torch.full_like(mask_pxl2d, float("+inf"))
    mask_pxl2d_masked[mask_pxl2d_masked_mask] = mask_pxl2d[mask_pxl2d_masked_mask]
    mask_verts2d_x_min = (
        mask_pxl2d_masked[..., 0].flatten(2).min(dim=-1).values * downsample_rate
    ).clamp(0, W - 1)
    mask_verts2d_y_min = (
        mask_pxl2d_masked[..., 1].flatten(2).min(dim=-1).values * downsample_rate
    ).clamp(0, H - 1)

    # B x C x 4
    scales = torch.stack(
        [
            (mesh_verts2d_x_max - cx) / (mask_verts2d_x_max - cx),
            (mesh_verts2d_x_min - cx) / (mask_verts2d_x_min - cx),
            (mesh_verts2d_y_max - cy) / (mask_verts2d_y_max - cy),
            (mesh_verts2d_y_min - cy) / (mask_verts2d_y_min - cy),
        ],
        dim=-1,
    )

    if multiview:
        scale = (
            scales.permute(1, 0, 2).flatten(1).median(dim=-1).values[None,].repeat(B, 1)
        )
    else:
        scale = scales.flatten(2).median(dim=-1).values

    if (scale < 0.01).any() or (scale > 100.0).any():
        scale[scale < 0.01] = 1.0
        scale[scale > 100.0] = 1.0
        logger.warning("setting scale of <0.01 or >100. to 1.")

    depth = b_cams_multiview_tform4x4_obj[:, :, 2, 3] * scale

    return depth
