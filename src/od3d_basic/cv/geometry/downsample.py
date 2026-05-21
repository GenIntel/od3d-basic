import logging
import torch

logger = logging.getLogger(__name__)

def fps(pts3d, K, fill=True):
    sample_count = min(len(pts3d), K)
    pts_inds = torch.zeros((K,), dtype=torch.long, device=pts3d.device)
    pts_vals = torch.zeros((K, 3), device=pts3d.device)

    if len(pts3d) > 0:
        pts_vals[:] = pts3d[0]

    # from pytorch3d.ops import sample_farthest_points
    # pts_vals_sampled, pts_inds_sampled = sample_farthest_points(points=pts3d[None,], K=K)
    # pts_vals_sampled = pts_vals_sampled[0]
    # pts_inds_sampled = pts_inds_sampled[0]

    from torch_cluster import fps

    # pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
    pts_inds_sampled = fps(pts3d, ratio=sample_count / len(pts3d))
    pts_vals_sampled = pts3d[pts_inds_sampled]

    sample_count = min(sample_count, len(pts_inds_sampled))

    if fill:
        pts_vals[:sample_count] = pts_vals_sampled[:sample_count]
        pts_inds[:sample_count] = pts_inds_sampled[:sample_count]
    else:
        pts_vals = pts_vals_sampled[:sample_count]
        pts_inds = pts_inds_sampled[:sample_count]
    return pts_inds, pts_vals


def farthest_point_sampling(pts3d_cls, K):
    import pytorch3d.ops

    pts3d_cls, _ = pytorch3d.ops.sample_farthest_points(pts3d_cls[None,], K=K)
    pts3d_cls = pts3d_cls[0]
    return pts3d_cls


def voxel_downsampling(
    pts3d_cls,
    K,
    top_bins_perc=1.0,
    return_mask=False,
    return_voxel_grid=False,
    min_steps=1,
):
    """
    Args:
        pts3d_cls (torch.Tensor): Nx3

    Returns:
        pts3d_cls (torch.Tensor): (<=K)x3
    """
    device = pts3d_cls.device
    bounds_max, _ = pts3d_cls.max(dim=-2)
    bounds_min, _ = pts3d_cls.min(dim=-2)
    bounds = bounds_max - bounds_min
    voxel_size = (bounds.prod(dim=-1) / K) ** (1 / 3)
    steps = (bounds / voxel_size).int()
    steps[steps < min_steps] = min_steps
    pts3d_voxel = torch.stack(
        torch.meshgrid(
            torch.linspace(
                start=bounds_min[0],
                end=bounds_max[0],
                steps=steps[0],
                device=device,
            ),
            torch.linspace(
                start=bounds_min[1],
                end=bounds_max[1],
                steps=steps[1],
                device=device,
            ),
            torch.linspace(
                start=bounds_min[2],
                end=bounds_max[2],
                steps=steps[2],
                device=device,
            ),
            indexing="xy",
        ),
        dim=-1,
    )
    # dist = (pts3d_voxel.reshape(-1, 3)[None, :] - pts3d_cls[:, None, ]).norm(dim=-1)
    dist = torch.cdist(pts3d_voxel.reshape(-1, 3)[None,], pts3d_cls[None,])[0]

    _, dist_min_ids_from_pts_to_voxels = dist.min(dim=-2)
    voxel_ids, voxel_counts = dist_min_ids_from_pts_to_voxels.unique(return_counts=True)
    voxel_ids = voxel_ids[
        voxel_counts.argsort(descending=True)[: int(len(voxel_ids) * top_bins_perc)]
    ]

    if return_voxel_grid:
        occ_grid = torch.zeros(size=pts3d_voxel.shape[:3]).to(
            device=device,
            dtype=torch.bool,
        )
        occ_grid.view(-1)[voxel_ids] = True
        occ_grid = occ_grid  # .permute(2, 1, 0)
        occ_grid_range = bounds
        occ_grid_offset = bounds_min
        # occ_grid_offset = occ_grid_offset.flip(dims=(0,))
        # occ_grid_range = occ_grid_range.flip(dims=(0,))
        return occ_grid, occ_grid_range, occ_grid_offset

    _, dist_min_ids_from_voxel_to_pts = dist.min(dim=-1)
    pts3d_ids = dist_min_ids_from_voxel_to_pts[voxel_ids]

    pts3d_mask = torch.zeros(size=(pts3d_cls.shape[0],)).to(
        dtype=torch.bool,
        device=device,
    )
    pts3d_mask[pts3d_ids] = True

    pts3d_cls = pts3d_cls[pts3d_mask]
    # del pts3d_voxel
    # del dist
    # del dist_min_ids_from_voxel_to_pts

    if return_mask:
        return pts3d_cls, pts3d_mask
    else:
        return pts3d_cls


def random_sampling(pts3d_cls, pts3d_max_count, return_mask=False):
    sample_ids = torch.randperm(pts3d_cls.shape[0])
    pts3d_mask = torch.zeros(size=(pts3d_cls.shape[0],)).to(
        dtype=torch.bool,
        device=pts3d_cls.device,
    )
    pts3d_mask[sample_ids[:pts3d_max_count]] = True
    pts3d_cls = pts3d_cls[pts3d_mask]

    if return_mask:
        return pts3d_cls, pts3d_mask
    else:
        return pts3d_cls

def random_sampling_with_fill(pts3d_cls, pts3d_count, return_ids=False):
    sample_ids = torch.randperm(pts3d_cls.shape[0])
    if sample_ids.shape[0] == 0:
        logger.info('no pts3d available to sample from...')
    elif sample_ids.shape[0] < pts3d_count:
        sample_ids = sample_ids.repeat((pts3d_count // sample_ids.shape[0]) + 1)[:pts3d_count]
    else:
        sample_ids = sample_ids[:pts3d_count]

    pts3d_cls = pts3d_cls[sample_ids]

    if return_ids:
        return pts3d_cls, sample_ids
    else:
        return pts3d_cls