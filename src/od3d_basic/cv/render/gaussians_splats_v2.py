import logging

logger = logging.getLogger(__name__)
import torch
from od3d_basic.cv.geometry.transform import proj3d2d_broadcast, transf3d_broadcast
from typing import Union


def rasterize_gaussians(
    cams_tform4x4_obj: torch.Tensor,
    cams_intr4x4: torch.Tensor,
    imgs_size: torch.Tensor,
    pts3d: torch.Tensor,
    pts3d_mask: torch.Tensor,
    pts3d_size: Union[float, torch.Tensor] = 0.01,
    # pts3d_size_rel_to_neighbor_dist: float = 0.5,
    pts3d_opacity: Union[float, torch.Tensor] = 1.0,
    pts3d_rotation: torch.Tensor = None,
    z_far: float = 10000.0,
    z_near: float = 0.01,
    topK: int = 3,
):
    """
    Args:
        cams_tform4x4_obj (torch.Tensor): B x 4 x 4,
        cams_intr4x4 (torch.Tensor): B x 4 x 4,
        imgs_size (torch.Tensor): 2, (height, width)
        pts3d (torch.Tensor): B x N x 3
        pts3d_mask (torch.Tensor): B x N
        pts3d_size (Union[float, torch.Tensor]): float or B x N  or B x N x 3
        pts3d_opacity (float)
        z_far (float)
        z_near (float)
        feats_dim_base (int)
    Returns:
        px_to_gs_id (torch.Tensor): B x G x H x W
        px_to_gs_opacity (torch.Tensor): B x G x H x W
        gs_depth (torch.Tensor): B x G
        gs_normal (torch.Tensor): B x G x 3
    """

    # logger.info(pts3d.shape)
    device = pts3d.device
    dtype = pts3d.dtype
    B = pts3d.shape[0]
    N_max = pts3d.shape[1]

    image_height, image_width = imgs_size
    image_height = int(image_height)
    image_width = int(image_width)

    gs_ids = []
    px_to_gs_id = []
    px_to_gs_opacity = []
    gs_depth = []
    gs_normal = []

    for b in range(B):
        N = int(pts3d_mask[b].sum())
        cam_tform4x4_obj_b = cams_tform4x4_obj[b]
        cam_intr4x4_b = cams_intr4x4[b]
        pts3d_b = pts3d[b, pts3d_mask[b]]  # .clone()
        if isinstance(pts3d_opacity, torch.Tensor):
            pts3d_opacity_b = pts3d_opacity[b, pts3d_mask[b]]  # .clone()
        else:  # pts3d_opacity is float
            pts3d_opacity_b = torch.Tensor([pts3d_opacity]).to(device=device)

        if pts3d_rotation is not None:
            pts3d_rotation_b = pts3d_rotation[b, pts3d_mask[b]]  # .clone()
        else:
            pts3d_rotation_b = None

        if cam_tform4x4_obj_b.isnan().any():
            logger.info(f"cam_tform4x4_obj contains NaN: {cam_tform4x4_obj_b}")

        pts3d_b_cam = transf3d_broadcast(pts3d=pts3d_b, transf4x4=cam_tform4x4_obj_b)

        if pts3d_b.isnan().any():
            logger.info(f"pts3d contains NaN: {pts3d_b}")

        # pts3d_b_dists = torch.cdist(
        #     pts3d_b_cam.clone().detach(),
        #     pts3d_b_cam.clone().detach(),
        # )
        # pts3d_b_dists.fill_diagonal_(torch.inf)
        # pts3d_b_dists = pts3d_b_dists.clamp(1e-5, 1e5)

        # pts3d_size_b = (
        #     pts3d_b_dists.min(dim=-1)
        #     .values[:, None]
        #     .mean(dim=0, keepdim=True)
        #     .expand(N, 3)
        #     * pts3d_size_rel_to_neighbor_dist
        # )

        if isinstance(pts3d_size, float):
            pts3d_size_b = torch.ones_like(pts3d_b) * pts3d_size
        else:  # if pts3d_size is torch.Tensor:
            pts3d_size_b = pts3d_size[b, pts3d_mask[b]]
        gaussians_z_fov = (pts3d_b_cam[:, 2] > z_near) * (pts3d_b_cam[:, 2] < z_far)

        # pts3d_size_b = pts3d_size # .clone()

        # pts3d_size_b = pts3d_b_dists.min(dim=-1).values[:, None].expand(N, 3) pts3d_size_rel_to_neighbor_dist

        # pts3d_size_b = pts3d_size_b.clamp(1e-5, 1e5)  # otherwise illegal access memory

        means2d = proj3d2d_broadcast(pts3d=pts3d_b_cam, proj4x4=cam_intr4x4_b)

        # print(means2d[:10])

        # 2 x H x W
        grid_pxl2d = torch.stack(
            torch.meshgrid(
                torch.arange(image_height),
                torch.arange(image_width),
                indexing="xy",
            ),
            dim=0,
        )

        # N x 2 x 3
        cov3d_var = pts3d_size_b**2
        if pts3d_rotation_b is None:
            cov3d_rot = (torch.eye(3).to(device, dtype))[None,].repeat(N, 1, 1)
        else:
            from pytorch3d.transforms import quaternion_to_matrix

            cov3d_rot = quaternion_to_matrix(pts3d_rotation_b)

        cov3d = (
            cov3d_rot
            * cov3d_var[
                :,
                :,
                None,
            ]
        )

        if cov3d_var.isnan().any():
            logger.info(f"cov3d_var contains NaN: {cov3d_var.isnan().any()}")

        if cov3d_rot.isnan().any():
            logger.info(f"cov3d_rot contains NaN: {cov3d_rot.isnan().any()}")

        jacobian3d2d = torch.zeros((N, 2, 3)).to(device, dtype)
        # J_K = [
        #   fx/z, 0, -fx*x/z^2;
        #   0, fy/z, -fy*y/z^2
        #   ]
        jacobian3d2d[:, 0, 0] = cam_intr4x4_b[0, 0] / (pts3d_b_cam[:, 2].abs() + 1e-10)
        jacobian3d2d[:, 0, 2] = (
            -cam_intr4x4_b[0, 0] * pts3d_b_cam[:, 0] / (pts3d_b_cam[:, 2] ** 2 + 1e-10)
        )
        jacobian3d2d[:, 1, 1] = cam_intr4x4_b[1, 1] / (pts3d_b_cam[:, 2].abs() + 1e-10)
        jacobian3d2d[:, 1, 2] = (
            -cam_intr4x4_b[1, 1] * pts3d_b_cam[:, 1] / (pts3d_b_cam[:, 2] ** 2 + 1e-10)
        )

        if jacobian3d2d.isnan().any():
            logger.info(f"jacobian contains NaN: {jacobian3d2d.isnan().any()}")
            # logger.info(f'pts3d are {pts3d_b_cam}')
            logger.info(f"cam_tform4x4_obj is {cam_tform4x4_obj_b}")

        cov2d = jacobian3d2d @ cov3d @ jacobian3d2d.permute(0, 2, 1)
        cov2d[~gaussians_z_fov, :, :] = 0.0
        cov2d[~gaussians_z_fov, 0, 0] = 1.0
        cov2d[~gaussians_z_fov, 1, 1] = 1.0
        # note: constant 2d covariance for debug
        # cov2d = (torch.eye(2).to(device, dtype))[None,].repeat(N, 1, 1) * 5

        inv_cov2d = torch.inverse(cov2d)

        if inv_cov2d.isnan().any():
            logger.info(f"inv_cov2d contains NaN: {inv_cov2d.isnan().any()}")

        # note: visualization for debug
        # from od3d_basic.cv.visual.show import show_scene2d, show_img, show_imgs
        # show_scene2d(pts2d =[means2d[:100], grid_pxl2d.flatten(1).permute(1,0)[:]])

        grid_pxl2d = grid_pxl2d.to(dtype=dtype, device=device)
        grid_pxl2d_dist2d = (
            grid_pxl2d[:, :, :, None] - means2d.permute(1, 0)[:, None, None]
        ).abs()

        grid_pxl2d_cov_dist = (
            (grid_pxl2d_dist2d[0] ** 2) * inv_cov2d[None, None, :, 0, 0]
            + (grid_pxl2d_dist2d[1] ** 2) * inv_cov2d[None, None, :, 1, 1]
            + 2
            * grid_pxl2d_dist2d[0]
            * grid_pxl2d_dist2d[1]
            * inv_cov2d[None, None, :, 0, 1]
        )

        if grid_pxl2d_cov_dist.isnan().any():
            logger.info(
                f"grid_pxl2d_cov_dist contains NaN: {grid_pxl2d_cov_dist.isnan().any()}",
            )

        grid_pxl2d_cov_opacity = pts3d_opacity_b[None, None] * torch.exp(
            -0.5 * grid_pxl2d_cov_dist,
        )

        # grid_pxl2d_cov_opacity[grid_pxl2d_cov_opacity < 0.5] = 0. # note: neglect smoothing

        if grid_pxl2d_cov_opacity.isnan().any():
            logger.info(
                f"(1) grid_pxl2d_cov_opacity contains NaN: {grid_pxl2d_cov_opacity.isnan().any()}",
            )

        grid_pxl2d_cov_opacity *= gaussians_z_fov
        if grid_pxl2d_cov_opacity.isnan().any():
            logger.info(
                f"(2) grid_pxl2d_cov_opacity contains NaN: {grid_pxl2d_cov_opacity.isnan().any()}",
            )

        pts3d_sorted_id = pts3d_b_cam[:, 2].argsort(descending=False)
        grid_pxl2d_cov_opacity = grid_pxl2d_cov_opacity[:, :, pts3d_sorted_id]

        pts3d_b_mask_cumulative = (~pts3d_mask[b]).cumsum(dim=-1)
        gs_ids_b_sorted = pts3d_sorted_id + pts3d_b_mask_cumulative[pts3d_sorted_id]

        from od3d_basic.cv.select import append_const_front

        grid_pxl2d_cov_opacity_append_zeros_front = append_const_front(
            grid_pxl2d_cov_opacity,
            dim=-1,
            value=0.0,
        )[:, :, :-1]
        grid_pxl2d_opacity = grid_pxl2d_cov_opacity * (
            (1.0 - grid_pxl2d_cov_opacity_append_zeros_front).cumprod(dim=-1)
        )

        if grid_pxl2d_opacity.isnan().any():
            logger.info(
                f"(1) grid_pxl2d_opacity contains NaN: {grid_pxl2d_opacity.isnan().any()}",
            )

        if topK is not None:
            px_to_gs_id_b_sorted = grid_pxl2d_opacity.argsort(descending=True, dim=-1)[
                ...,
                :topK,
            ]
            px_to_gs_id_b = gs_ids_b_sorted[px_to_gs_id_b_sorted]
            grid_pxl2d_opacity_total = grid_pxl2d_opacity.detach().sum(
                dim=-1,
                keepdim=True,
            )
            grid_pxl2d_opacity = torch.gather(
                input=grid_pxl2d_opacity,
                dim=-1,
                index=px_to_gs_id_b_sorted,
            )

            if grid_pxl2d_opacity.isnan().any():
                logger.info(
                    f"topK(1) grid_pxl2d_opacity contains NaN: {grid_pxl2d_opacity.isnan().any()}",
                )

            grid_pxl2d_opacity_partial = grid_pxl2d_opacity.detach().sum(
                dim=-1,
                keepdim=True,
            )
            grid_pxl2d_opacity *= (
                grid_pxl2d_opacity_total / (grid_pxl2d_opacity_partial + 1e-10)
            ).clamp(0, 10)

            if grid_pxl2d_opacity.isnan().any():
                logger.info(
                    f"topK(2) grid_pxl2d_opacity contains NaN: {grid_pxl2d_opacity.isnan().any()}",
                )

            # original

        # note: visualization for debug
        # from od3d_basic.cv.visual.show import show_scene2d, show_img, show_imgs
        # show_img(grid_pxl2d_opacity.sum(dim=-1)[None,].clamp(0, 1))

        # img = torch.einsum("hwn,nc->chw", grid_pxl2d_opacity, feats_b)

        gs_ids.append(gs_ids_b_sorted)  # N,
        if topK is not None:
            px_to_gs_id.append(px_to_gs_id_b)

        px_to_gs_opacity.append(grid_pxl2d_opacity)  # HxWxN (sorted)

        if grid_pxl2d_opacity.isnan().any():
            logger.info(
                f"(2) grid_pxl2d_opacity contains NaN: {grid_pxl2d_opacity.isnan().any()}",
            )

        gs_depth.append(pts3d_b_cam[:, 2])  # N (not sorted)

        # cam_center = cam_centers_from_tform4x4(cam_tform4x4_obj_b[:, :3])
        # verts3d = self.get_verts_ncds_with_mesh_id(mesh_id)
        # view_points = cam_center - verts3d
        # view_points = view_points / view_points.norm(dim=-1, keepdim=True)
        # gs_normal.append(cam_tform4x4_obj_b[:, :3])
        # logger.info(px_to_gs_opacity[-1].shape)

    # logger.info(len(px_to_gs_opacity))

    px_to_gs_opacity = torch.stack(px_to_gs_opacity, dim=0)
    gs_depth = torch.stack(gs_depth, dim=0)
    # gs_normal = torch.stack(gs_normal, dim=0)

    if px_to_gs_opacity.isnan().any():
        logger.info(f"px_to_gs_opacity contains NaN: {px_to_gs_opacity.isnan().any()}")

    if topK is not None:
        gs_ids = torch.stack(px_to_gs_id, dim=0)
    else:
        gs_ids = torch.stack(gs_ids, dim=0)

    return gs_ids, px_to_gs_opacity, gs_depth, gs_normal


def blend_gaussians(
    px_to_gs_opacity: torch.Tensor,
    feats: torch.Tensor,
    feat_bg: torch.Tensor = None,
):
    """
    Args:
        px_to_gs_opacity (torch.Tensor): B x N x H x W
        feats (torch.Tensor): B x N x F / B x N x H x W x F
    Returns:
        img (torch.Tensor): B x F x H x W
    """
    if feat_bg is not None:
        if feats.dim() == 3:
            feats = torch.cat(
                [feats, feat_bg[None, None].expand(feats.shape[0], -1, -1)],
                dim=1,
            )
        else:
            feat_bg_ext = feat_bg.clone()[None, None, None, None].expand(
                feats.shape[0],
                -1,
                feats.shape[2],
                feats.shape[3],
                -1,
            )
            feats = torch.cat([feats, feat_bg_ext], dim=1)
        px_to_gs_opacity = torch.cat(
            [px_to_gs_opacity, 1.0 - px_to_gs_opacity.sum(dim=-1, keepdim=True)],
            dim=-1,
        )

    if feats.dim() == 3:
        imgs = torch.einsum("bhwn,bnc->bchw", px_to_gs_opacity, feats)
    else:
        imgs = torch.einsum("bhwn,bnhwc->bchw", px_to_gs_opacity, feats)

    return imgs


def render_gaussians(
    cams_tform4x4_obj: torch.Tensor,
    cams_intr4x4: torch.Tensor,
    imgs_size: torch.Tensor,
    pts3d: torch.Tensor,
    feats: torch.Tensor,
    pts3d_size: float = 0.5,
    opacity: float = 1.0,
    z_far: float = 10000.0,
    z_near: float = 0.01,
    pts3d_mask: torch.Tensor = None,
    feat_bg: torch.Tensor = None,
    topK: int = 3,
):
    """
    Args:
        cams_tform4x4_obj (torch.Tensor): B x 4 x 4,
        cams_intr4x4 (torch.Tensor): B x 4 x 4,
        imgs_size (torch.Tensor): 2, (height, width)
        pts3d (torch.Tensor): B x N x 3
        pts3d_mask (torch.Tensor): B x N
        feats (torch.Tensor): B x N x F
        opacity (float)
        z_far (float)
        z_near (float)
        feats_dim_base (int)
    Returns:
        img (torch.Tensor): B x F x H x W
    """
    if pts3d_mask is None:
        pts3d_mask = torch.ones_like(pts3d[..., 0]).to(dtype=bool)

    px_to_gs_ids, px_to_gs_opacity, gs_depth, gs_normal = rasterize_gaussians(
        cams_tform4x4_obj=cams_tform4x4_obj,
        cams_intr4x4=cams_intr4x4,
        imgs_size=imgs_size,
        pts3d=pts3d,
        pts3d_mask=pts3d_mask,
        pts3d_size=pts3d_size,
        pts3d_opacity=opacity,
        z_far=z_far,
        z_near=z_near,
        topK=topK,
    )

    return select_and_blend_gaussians(
        px_to_gs_ids,
        px_to_gs_opacity,
        feats,
        feat_bg=feat_bg,
    )


def select_and_blend_gaussians(
    gs_ids: torch.Tensor,
    px_to_gs_opacity: torch.Tensor,
    feats: torch.Tensor,
    feat_bg: torch.Tensor = None,
):
    """
    Args:
        gs_ids (torch.Tensor): B x G'
        px_to_gs_opacity (torch.Tensor): B x G' x H x W
        feats (torch.Tensor): B x G x F
    Returns:
        img (torch.Tensor): B x F x H x W
    """

    # gs_ids: B x G' / B x H x W x G'
    # feats: B x G x F
    # feats_selected: B x G' x F / B x G' x H x W x F

    if gs_ids.dim() == 2:
        feats_selected = torch.gather(
            feats,
            1,
            gs_ids[:, :, None].expand(-1, -1, feats.shape[-1]),
        )
    else:
        feats_selected = torch.gather(
            feats[:, :, None, None].expand(
                -1,
                -1,
                gs_ids.shape[1],
                gs_ids.shape[2],
                -1,
            ),
            1,
            gs_ids.permute(0, 3, 1, 2)[..., None].expand(
                -1,
                -1,
                -1,
                -1,
                feats.shape[-1],
            ),
        )
    return blend_gaussians(
        px_to_gs_opacity=px_to_gs_opacity,
        feats=feats_selected,
        feat_bg=feat_bg,
    )
