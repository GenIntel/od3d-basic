import torch
import numpy as np
# NOCS, NOCS_0C, NOCS3D, NOCS3D_0C, NOCS_DIAG_0C

def nocs_0c_to_nocs(nocs_0c):
    return ((nocs_0c + 1).clamp(0, 2)) / 2.

def nocs_to_nocs_0c(nocs):
    return (nocs - 0.5).clamp(-1, 1)

def nocs_0c_to_rgb(nocs_0c, resolution=5):
    nocs = nocs_0c_to_nocs(nocs_0c)
    return nocs_to_rgb(nocs=nocs, resolution=resolution)

def pastel_map(coords: torch.Tensor) -> torch.Tensor:
    """
    Map 3D coordinates in [0, 1]^3 to pleasant pastel RGB colors.

    Args:
        coords (torch.Tensor): shape (..., 3), values in [0, 1]
    Returns:
        torch.Tensor: same shape, RGB values in [0, 1]
    """
    # Optional nonlinear warping to spread coordinates
    #x = torch.sin(coords * torch.pi / 2)

    # Mix coordinates nonlinearly to add variation but keep smoothness
    #r = 0.4 + 0.6 * (0.6 * x[..., 0] + 0.2 * x[..., 1] + 0.2 * x[..., 2])
    #g = 0.4 + 0.6 * (0.2 * x[..., 0] + 0.6 * x[..., 1] + 0.2 * x[..., 2])
    #b = 0.4 + 0.6 * (0.2 * x[..., 0] + 0.2 * x[..., 1] + 0.6 * x[..., 2])
    #rgb = torch.stack([r, g, b], dim=-1)

    rgb = coords * 0.6 + 0.4
    # Slight desaturation (brings to pastel range)
    rgb_mean = rgb.mean(dim=-1, keepdim=True)
    pastel_rgb = 0.99 * rgb + 0.01 * rgb_mean

    # Ensure within [0, 1]
    return torch.clamp(pastel_rgb, 0, 1)

def nocs_to_rgb(nocs, resolution=5, nocs_dim=-1):
    """Combine HSV golden ratio with horizontal checkerboard pattern.
    Uses HSV colors but modulates brightness based on horizontal (z-axis) layers."""

    nocs = nocs.clone()

    if nocs_dim != -1:
        nocs = nocs.movedim(nocs_dim, -1)

    #nocs = nocs[..., [0, 1, 2]] # default (left-right ambiguity) (omni6dpose)
    #nocs = nocs[..., [1, 2, 0]] # default (back-front ambiguity) (omni6dpose)
    nocs = nocs[..., [2, 0, 1]] # default 


    nocs = torch.clamp(nocs, 0.0, 1.0)
    voxel_size = 1.0 / resolution
    idx = torch.floor(nocs / voxel_size)
    idx = torch.clamp(idx, 0, resolution - 1).long()

    num_voxels = resolution ** 3

    # Distribute hues evenly using golden ratio (same as HSV method)
    hues = np.linspace(0, 1, num_voxels, endpoint=False)
    golden_ratio = (1 + np.sqrt(5)) / 2
    hues = (hues * golden_ratio) % 1.0

    # Create checkerboard pattern based on z-index (horizontal layers)
    saturations = np.ones(num_voxels) * 0.8
    values = np.zeros(num_voxels)

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                linear_idx = i * resolution * resolution + j * resolution + k
                # Alternate brightness based on z-layer (k index)
                if k % 2 == 0:
                    values[linear_idx] = 0.9  # Bright
                else:
                    values[linear_idx] = 0.6  # Darker

    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    hsv_colors = np.stack([hues, saturations, values], axis=-1)
    rgb_colors = hsv_to_rgb(hsv_colors.reshape(1, -1, 3)).reshape(-1, 3)

    palette = torch.from_numpy(rgb_colors).float()
    palette = palette.to(device=nocs.device, dtype=nocs.dtype)

    # palette = pastel_map(palette)

    palette[0, :] = 1.  # background

    linear_idx = (
            idx[..., 0] * resolution * resolution
            + idx[..., 1] * resolution
            + idx[..., 2]
    )
    cube_nocs = palette.index_select(0, linear_idx.view(-1)).view_as(nocs)


    cube_nocs = cube_nocs.type_as(nocs)

    if nocs_dim != -1:
        cube_nocs = cube_nocs.movedim(-1, nocs_dim)
    return cube_nocs

def nocs_to_rgb_v1(nocs, resolution=5, nocs_dim=-1):
    """Convert continuous NOCS coordinates in [0, 1] into a discrete voxelized
    Cube-NOCS representation. Each voxel receives a deterministic, yet randomly
    generated RGB color that depends only on the chosen resolution. This makes
    the visualization repeatable across runs while keeping neighboring voxels
    easy to distinguish."""

    nocs = nocs.clone()

    if nocs_dim != -1:
        nocs = nocs.movedim(nocs_dim, -1)

    # Clamp NOCS into valid range
    nocs = torch.clamp(nocs, 0.0, 1.0)

    # Discretize each axis using integer binning
    voxel_size = 1.0 / resolution
    idx = torch.floor(nocs / voxel_size)
    idx = torch.clamp(idx, 0, resolution - 1).long()

    # Pre-compute deterministic voxel color palette; cache per resolution
    palette = None #  self._cube_nocs_palette_cache.get(resolution)
    if palette is None:
        generator = torch.Generator(device="cpu")
        # Deterministic seed based solely on resolution (large prime multiplier
        # helps decorrelate adjacent resolutions).
        generator.manual_seed(resolution * 104729 + 8191)
        palette = torch.rand(
            (resolution ** 3, 3), generator=generator, dtype=torch.float32
        )
        palette = pastel_map(palette)

        palette[0, :] = 1.  # background
        # self._cube_nocs_palette_cache[resolution] = palette

    palette = palette.to(device=nocs.device, dtype=nocs.dtype)

    # Convert 3D voxel coordinate to linear index and fetch deterministic color
    linear_idx = (
        idx[..., 0] * resolution * resolution
        + idx[..., 1] * resolution
        + idx[..., 2]
    )
    cube_nocs = palette.index_select(0, linear_idx.view(-1)).view_as(nocs)

    cube_nocs = cube_nocs.type_as(nocs)

    if nocs_dim != -1:
        cube_nocs = cube_nocs.movedim(-1, nocs_dim)
    return cube_nocs

def voxel_color(p):
    # p : [0, 1]^3 -> 0-1^3
    voxel_size = 0.2
    rep = 3
    grid_size = 9
    cell = (torch.floor(grid_size * p) % rep) / (rep - 1)
    #p_voxel = (cell + 0.5) * voxel_size
    #  p_voxel_at_z0 = project_along_111_to_z0(p_voxel)
    # cell = torch.floor((1./voxel_size) * p_voxel_at_z0)

    return cell * 1.

