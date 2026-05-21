import logging

logger = logging.getLogger(__name__)

import torch

from od3d_basic.cv.select import batched_indexMD_select, batched_index_select


def batch_point_face_distance(
    verts1,
    faces1,
    pts2,
    verts1_mask=None,
    faces1_mask=None,
    pts2_mask=None,
):
    """
    Args:
        verts1 (torch.Tensor): (B, N, 3)
        faces1 (torch.LongTensor): (B, F, 3)
        pts2 (torch.Tensor): (B, M, 3)
        verts1_mask (torch.Tensor): (B, N)
        faces1_mask (torch.Tensor): (B, F)
        pts2_mask (torch.Tensor): (B, M)
    Returns:
        chamfer_dist (torch.Tensor): (B,)
    """

    from pytorch3d.loss.point_mesh_distance import point_mesh_face_distance
    from pytorch3d.structures.meshes import Meshes as PT3DMeshes
    from pytorch3d.structures.pointclouds import Pointclouds as PT3DPCLs

    B = verts1.shape[0]
    if verts1_mask is None:
        pt3d_verts1 = [verts1[b] for b in range(B)]
    else:
        pt3d_verts1 = [verts1[b, verts1_mask[b]] for b in range(B)]

    if faces1_mask is None:
        pt3d_faces1 = [faces1[b] for b in range(B)]
    else:
        pt3d_faces1 = [faces1[b, faces1_mask[b]] for b in range(B)]

    pt3d_meshes = PT3DMeshes(
        verts=pt3d_verts1,
        faces=pt3d_faces1,
    )

    if pts2_mask is None:
        pt3d_pts2 = [pts2[b] for b in range(B)]
    else:
        pt3d_pts2 = [pts2[b, pts2_mask[b]] for b in range(B)]

    pt3d_pcls = PT3DPCLs(points=pt3d_pts2)

    loss = point_mesh_face_distance(meshes=pt3d_meshes, pcls=pt3d_pcls)
    """
    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    return loss


def get_nearest_neighbor(
    pts1,
    pts2,
    pts1_mask=None,
    pts2_mask=None,
    only_pts2_nn=False,
    replace_dist_inf_with_m1 = True,
):
    """
    Args:
        pts1 (torch.Tensor): (BF1, ... BL1, N, 3)
        pts2 (torch.Tensor): (BF2, ..., BL2, M, 3)
        pts1_mask (torch.Tensor): (BF1, ... BL1, N)
        pts2_mask (torch.Tensor): (BF1, ... BL1, M)
    Returns:
        argmin_pts1_from_pts2 (torch.Tensor): (BF1, ... BL1, M)
        argmin_pts2_from_pts1 (torch.Tensor): (BF1, ... BL1, N)
    """
    #pts1_mask = pred_meshes_verts_mask
    #pts2_mask = gt_meshes_verts_mask
    pts1_batch_shape = pts1.shape[:-2]
    pts2_batch_shape = pts2.shape[:-2]

    pts1 = pts1.reshape(-1, *pts1.shape[-2:])
    pts2 = pts2.reshape(-1, *pts2.shape[-2:])
    device = pts1.device
    dtype = pts1.dtype

    if pts1_mask is None:
        pts1_mask = torch.ones_like(pts1[..., 0], dtype=torch.bool)
    if pts2_mask is None:
        pts2_mask = torch.ones_like(pts2[..., 0], dtype=torch.bool)

    pts1_mask = pts1_mask.reshape(-1, *pts1_mask.shape[-1:])
    pts2_mask = pts2_mask.reshape(-1, *pts2_mask.shape[-1:])

    pairs_mask = pts1_mask[:, :, None] * pts2_mask[:, None, :]
    verts_cdist_pts1_pts2 = torch.cdist(pts1, pts2)
    verts_cdist_pts1_pts2_min = (
        verts_cdist_pts1_pts2.detach().clone()
    )  # + ((~pairs_mask) * 1.) * torch.inf
    verts_cdist_pts1_pts2_min[~pairs_mask] = torch.inf

    argmin_pts1_from_pts2 = verts_cdist_pts1_pts2_min.argmin(dim=-2)  # BxG

    if replace_dist_inf_with_m1:
        #argmin_pts1_from_pts2[~pairs_mask] = -1
        pairs_pts1_from_pts2 = torch.stack(
            [
                argmin_pts1_from_pts2,
                torch.arange(argmin_pts1_from_pts2.shape[-1])
                .view(1, -1)
                .expand(argmin_pts1_from_pts2.shape)
                .to(
                    device=device,
                ),
            ],
            dim=-1,
        )
        chamfer_dist_pairwise = batched_indexMD_select(
            indexMD=pairs_pts1_from_pts2,
            inputMD=verts_cdist_pts1_pts2_min,
        )  # B x M+N
        argmin_pts1_from_pts2[chamfer_dist_pairwise == torch.inf] = -1

    argmin_pts1_from_pts2 = argmin_pts1_from_pts2.reshape(*pts2_batch_shape, -1)

    if only_pts2_nn:
        return argmin_pts1_from_pts2

    argmin_pts2_from_pts1 = verts_cdist_pts1_pts2_min.argmin(dim=-1)  # BxP

    if replace_dist_inf_with_m1:
        # argmin_pts2_from_pts1[~pairs_mask] = -1
        pairs_pts2_from_pts1 = torch.stack(
            [
                torch.arange(argmin_pts2_from_pts1.shape[-1])
                .view(1, -1)
                .expand(argmin_pts2_from_pts1.shape)
                .to(
                    device=device,
                ),
                argmin_pts2_from_pts1,
            ],
            dim=-1,
        )
        chamfer_dist_pairwise = batched_indexMD_select(
            indexMD=pairs_pts2_from_pts1,
            inputMD=verts_cdist_pts1_pts2_min,
        )  # B x M+N
        argmin_pts2_from_pts1[chamfer_dist_pairwise == torch.inf] = -1


    argmin_pts2_from_pts1 = argmin_pts2_from_pts1.reshape(*pts1_batch_shape, -1)




    return argmin_pts1_from_pts2, argmin_pts2_from_pts1

def batch_chamfer_distance(
    pts1,
    pts2,
    pts1_mask=None,
    pts2_mask=None,
    uniform_weight_pts1=True,
    only_pts2_nn=False,
):
    """
    Args:
        pts1 (torch.Tensor): (B, N, 3)
        pts2 (torch.Tensor): (B, M, 3)
        pts1_mask (torch.Tensor): (B, N)
        pts2_mask (torch.Tensor): (B, M)
    Returns:
        chamfer_dist (torch.Tensor): (B,)
    """
    #pts1_mask = pred_meshes_verts_mask
    #pts2_mask = gt_meshes_verts_mask
    device = pts1.device
    dtype = pts1.dtype

    if pts1_mask is None:
        pts1_mask = torch.ones_like(pts1[..., 0], dtype=torch.bool)
    if pts2_mask is None:
        pts2_mask = torch.ones_like(pts2[..., 0], dtype=torch.bool)

    pairs_mask = pts1_mask[:, :, None] * pts2_mask[:, None, :]
    verts_cdist_pred_gt = torch.cdist(pts1, pts2)
    verts_cdist_pred_gt_min = (
        verts_cdist_pred_gt.detach().clone()
    )  # + ((~pairs_mask) * 1.) * torch.inf
    verts_cdist_pred_gt_min[~pairs_mask] = torch.inf

    argmin_pred_from_gt = verts_cdist_pred_gt_min.argmin(dim=-2)  # BxG

    pairs_pred_from_gt = torch.stack(
        [
            argmin_pred_from_gt,
            torch.arange(argmin_pred_from_gt.shape[-1])
            .view(1, -1)
            .expand(argmin_pred_from_gt.shape)
            .to(
                device=device,
            ),
        ],
        dim=-1,
    )
    if not only_pts2_nn:
        argmin_gt_from_pred = verts_cdist_pred_gt_min.argmin(dim=-1)  # BxP

        pairs_gt_from_pred = torch.stack(
            [
                torch.arange(argmin_gt_from_pred.shape[-1])
                .view(1, -1)
                .expand(argmin_gt_from_pred.shape)
                .to(
                    device=device,
                ),
                argmin_gt_from_pred,
            ],
            dim=-1,
        )

        pairs_pred_gt = torch.cat(
            [pairs_pred_from_gt, pairs_gt_from_pred],
            dim=1,
        )  # B x M+N x 2
    else:
        pairs_pred_gt = pairs_pred_from_gt

    from od3d_basic.cv.select import batched_indexMD_select, batched_index_select

    chamfer_dist_pairwise = batched_indexMD_select(
        indexMD=pairs_pred_gt,
        inputMD=verts_cdist_pred_gt,
    )  # B x M+N

    M = pts2_mask.shape[-1]
    N = pts1_mask.shape[-1]
    B = pts1_mask.shape[0]

    if uniform_weight_pts1:
        pairs_pred_gt_ext = pairs_pred_gt.clone()
        pairs_pred_gt_ext[:, :M, 0][~pts2_mask] = N
        pairs_pred_gt_ext[:, M:, 0][~pts1_mask] = N
        pts1_counts = torch.nn.functional.one_hot(pairs_pred_gt_ext[:, :, 0]).sum(dim=1)
        pts1_counts = pts1_counts[:, :N]
        pts1_weights = 1.0 / pts1_counts
        pts1_weights[~pts1_mask] = 0.
        pts2_weights_from_pts1 = batched_index_select(
            index=pairs_pred_gt[:, :M, 0],
            input=pts1_weights,
        )
        pts2_mask = pts2_mask.clone() * pts2_weights_from_pts1
        pts1_mask = pts1_mask.clone() * pts1_weights

    chamfer_dist_mean_pred_from_gt = (chamfer_dist_pairwise[:, :M] * pts2_mask).sum(
        dim=-1,
    ) / (pts2_mask.sum(dim=-1) + 1e-10)

    if not only_pts2_nn:
        chamfer_dist_mean_gt_from_pred = (chamfer_dist_pairwise[:, M:] * pts1_mask).sum(
            dim=-1,
        ) / (pts1_mask.sum(dim=-1) + 1e-10)
        chamfer_dist = (
            chamfer_dist_mean_pred_from_gt + chamfer_dist_mean_gt_from_pred
        ) / 2.0
    else:
        chamfer_dist = chamfer_dist_mean_pred_from_gt

    return chamfer_dist


# def batch_point_face_distance_v2(pts3d, meshes, objects_ids, pts3d_mask = None,):
"""
Args:
    pts3d: BxPx3
    pts3d_mask: BxP
    meshes (Meshes)
    objects_ids: B
"""


def batch_point_face_distance_v2(
    verts1,
    faces1,
    pts2,
    verts1_mask=None,
    faces1_mask=None,
    pts2_mask=None,
):
    """
    Args:
        verts1 (torch.Tensor): (B, N, 3)
        faces1 (torch.LongTensor): (B, F, 3)
        pts2 (torch.Tensor): (B, M, 3)
        verts1_mask (torch.Tensor): (B, N)
        faces1_mask (torch.Tensor): (B, F)
        pts2_mask (torch.Tensor): (B, M)
    Returns:
        chamfer_dist (torch.Tensor): (B,)
    """

    dists = []
    for b in range(len(verts1)):
        pts = pts2[b]
        if pts2_mask is not None:
            pts = pts[pts2_mask[b]]

        faces = faces1[b]
        if faces1_mask is not None:
            faces = faces[faces1_mask[b]]

        verts = verts1[b]
        if verts1_mask is not None:
            verts = verts[verts1_mask[b]]

        dist = points_faces_dist(pts3d=pts, verts=verts, faces=faces)
        dists.append(dist)

    dists = torch.stack(dists, dim=0)
    return dists


def points_faces_dist(pts3d, verts, faces):
    """

    Args:
        pts3d: Px3
        verts: Vx3
        faces: Fx3
    Returns:
        dist: float
    """

    faces_normals = torch.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
        dim=-1,
    )
    faces_area = faces_normals.norm(dim=-1)
    faces_large_enough = (faces_area > 1e-10) * (faces_area < 1e10)

    faces_normals = faces_normals / (faces_normals.norm(dim=-1, keepdim=True) + 1e-15)

    pts3d_signed_dist_to_faces = torch.einsum(
        "fd,fd->f",
        faces_normals,
        verts[faces[:, 0]],
    )[
        None,
    ] - torch.einsum(
        "fd,pd->pf",
        faces_normals,
        pts3d,
    )

    pts3d_on_faces = (
        pts3d[:, None] + faces_normals[None,] * pts3d_signed_dist_to_faces[:, :, None]
    )

    verts0_pts_faces = verts[faces[:, 0]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)
    verts1_pts_faces = verts[faces[:, 1]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)
    verts2_pts_faces = verts[faces[:, 2]][None,].repeat(pts3d_on_faces.shape[0], 1, 1)

    edge01_pts_faces = verts1_pts_faces - verts0_pts_faces
    edge02_pts_faces = verts2_pts_faces - verts0_pts_faces
    edge12_pts_faces = verts2_pts_faces - verts1_pts_faces

    pts3d_signed_dist_to_edge01 = torch.einsum(
        "pfd,pfd->pf",
        edge01_pts_faces,
        pts3d_on_faces - verts0_pts_faces,
    )
    pts3d_signed_dist_to_edge02 = torch.einsum(
        "pfd,pfd->pf",
        edge02_pts_faces,
        pts3d_on_faces - verts0_pts_faces,
    )
    pts3d_signed_dist_to_edge12 = torch.einsum(
        "pfd,pfd->pf",
        edge12_pts_faces,
        pts3d_on_faces - verts1_pts_faces,
    )

    pts3d_signed_dist_to_edge01 = pts3d_signed_dist_to_edge01.clamp(0.0, 1.0)
    pts3d_signed_dist_to_edge02 = pts3d_signed_dist_to_edge02.clamp(0.0, 1.0)
    pts3d_signed_dist_to_edge12 = pts3d_signed_dist_to_edge12.clamp(0.0, 1.0)

    pts3d_on_edge01 = (
        verts0_pts_faces + edge01_pts_faces * pts3d_signed_dist_to_edge01[:, :, None]
    )
    pts3d_on_edge02 = (
        verts0_pts_faces + edge02_pts_faces * pts3d_signed_dist_to_edge02[:, :, None]
    )
    pts3d_on_edge12 = (
        verts1_pts_faces + edge12_pts_faces * pts3d_signed_dist_to_edge12[:, :, None]
    )

    A = torch.stack(
        [
            verts0_pts_faces.detach(),
            verts1_pts_faces.detach(),
            verts2_pts_faces.detach(),
        ],
        dim=-1,
    )

    # A_full_rank_mask = torch.linalg.matrix_rank(A) == 3
    # A_full_rank_mask = torch.ones_like(A[..., 0, 0], dtype=torch.bool, device=A.device)
    A_full_rank_mask = faces_large_enough[None,].repeat(A.shape[0], 1)

    A[~A_full_rank_mask] = torch.eye(3).to(device=A.device, dtype=A.dtype)
    B = pts3d_on_faces.detach()
    X = torch.linalg.solve(A, B)  # alpha beta gamma
    X[~A_full_rank_mask] = torch.Tensor([-1, -1, -1]).to(device=X.device, dtype=X.dtype)

    pts3d_on_faces_closest = (A @ X[..., None]).squeeze(
        -1,
    )  #  pts3d_on_faces.clone()  # * 0
    pts3d_on_faces_closest_v0 = (
        (X[:, :, 0] >= 0.0) * (X[:, :, 1] <= 0.0) * (X[:, :, 2] <= 0.0)
    )
    pts3d_on_faces_closest_v1 = (
        (X[:, :, 0] <= 0.0) * (X[:, :, 1] >= 0.0) * (X[:, :, 2] <= 0.0)
    )
    pts3d_on_faces_closest_v2 = (
        (X[:, :, 0] <= 0.0) * (X[:, :, 1] <= 0.0) * (X[:, :, 2] >= 0.0)
    )

    pts3d_on_faces_closest_e01 = (
        (X[:, :, 0] >= 0.0) * (X[:, :, 1] >= 0.0) * (X[:, :, 2] <= 0.0)
    )
    pts3d_on_faces_closest_e02 = (
        (X[:, :, 0] >= 0.0) * (X[:, :, 1] <= 0.0) * (X[:, :, 2] >= 0.0)
    )
    pts3d_on_faces_closest_e12 = (
        (X[:, :, 0] <= 0.0) * (X[:, :, 1] >= 0.0) * (X[:, :, 2] >= 0.0)
    )

    pts3d_on_faces_closest_inside = (
        (X[:, :, 0] >= 0.0) * (X[:, :, 1] >= 0.0) * (X[:, :, 2] >= 0.0)
    )
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v0
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v1
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_v2
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e01
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e02
    pts3d_on_faces_closest_inside += pts3d_on_faces_closest_e12

    pts3d_on_faces_closest[~pts3d_on_faces_closest_inside] = torch.inf  #  999999.

    pts3d_on_faces_closest[pts3d_on_faces_closest_v0] = verts0_pts_faces[
        pts3d_on_faces_closest_v0
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_v1] = verts1_pts_faces[
        pts3d_on_faces_closest_v1
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_v2] = verts2_pts_faces[
        pts3d_on_faces_closest_v2
    ]

    pts3d_on_faces_closest[pts3d_on_faces_closest_e01] = pts3d_on_edge01[
        pts3d_on_faces_closest_e01
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_e02] = pts3d_on_edge02[
        pts3d_on_faces_closest_e02
    ]
    pts3d_on_faces_closest[pts3d_on_faces_closest_e12] = pts3d_on_edge12[
        pts3d_on_faces_closest_e12
    ]

    from od3d_basic.cv.select import batched_index_select

    dist_pts3d_faces = (pts3d[:, None] - pts3d_on_faces_closest).norm(dim=-1)

    pts3d_closest_face_id = dist_pts3d_faces.min(dim=-1).indices
    pts3d_on_faces_closest_sel = batched_index_select(
        pts3d_on_faces_closest,
        index=pts3d_closest_face_id[:, None],
        dim=1,
    )[:, 0]

    pts3d_closest_to_face_id = dist_pts3d_faces.min(dim=-2).indices
    pts3d_to_faces_closest_sel = pts3d[pts3d_closest_to_face_id]
    pts3d_on_faces_closest_to_pts3d_sel = batched_index_select(
        pts3d_on_faces_closest.permute(1, 0, 2),
        index=pts3d_closest_to_face_id[:, None],
        dim=1,
    )[:, 0]

    pts3d_dists_pt_to_face = (pts3d - pts3d_on_faces_closest_sel).norm(dim=-1)
    # pts3d_dists_face_to_pt = (pts3d_to_faces_closest_sel - pts3d_on_faces_closest_to_pts3d_sel).norm(dim=-1)

    pts3d_dist = pts3d_dists_pt_to_face.mean()

    # from od3d_basic.cv.visual.show import show_scene
    # show_scene(pts3d=[verts, torch.zeros((0, 3)).to(pts3d.device), pts3d[:1000], pts3d_on_face_closest[:1000] ],
    #           lines3d=[torch.stack([pts3d[:1000], pts3d_on_face_closest[:1000]],dim=-2)])

    return pts3d_dist
