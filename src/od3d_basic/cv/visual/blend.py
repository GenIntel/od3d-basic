import torch


def rgb_to_range01(rgb):
    if rgb.dtype == torch.uint8:
        rgb = rgb.clone() * 1. / 255.
    else:
        if (rgb < 0).any() or (rgb > 1).any():
            rgb = (rgb.clone() - rgb.min()) / (rgb.max() - rgb.min())
    return rgb

def norm_to_range01(values):
    values = (values - values.min()) / (values.max() - values.min())
    return values

def blend_rgb(rgb1, rgb2, alpha1=0.5, alpha2=0.5):
    if rgb1.dtype == torch.bool or rgb1.dtype == torch.float:
        rgb1 = rgb_to_range01(rgb1)
        rgb1 = rgb1 * 255.0
    if rgb2.dtype == torch.bool or rgb2.dtype == torch.float:
        rgb2 = rgb_to_range01(rgb2)
        rgb2 = rgb2 * 255.0
    return (alpha1 * rgb1 + alpha2 * rgb2).clamp(0, 255).to(torch.uint8)


def gaussian_scatter_image(uv, rgb, H, W, sigma=1.0):
    """
    Args:
        uv: (N, 2) - float pixel coordinates (x, y)
        rgb: (N, C)
        H, W: int - output image size
        sigma: float - standard deviation of the Gaussian
        kernel_size: int - size of the Gaussian kernel (odd number)

    Returns:
        image: (C, H, W) - reconstructed image
    """
    device = uv.device
    # 2 x H x W
    grid_pxl2d = torch.stack(
        torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="xy",
        ),
        dim=0,
    )
    grid_pxl2d = grid_pxl2d.to(device)
    grid_pxl2d_dist = (
        (grid_pxl2d[0, :, :, None] - uv[None, None, :, 0]) ** 2
        + (grid_pxl2d[1, :, :, None] - uv[None, None, :, 1]) ** 2
    ) ** 0.5
    grid_pxl2d_weight = torch.exp(-0.5 * (grid_pxl2d_dist / sigma) ** 2)
    image = grid_pxl2d_weight[..., None] * rgb[None, None]
    image = image.sum(dim=-2) / (grid_pxl2d_weight[..., None].sum(dim=-2) + 1e-10)
    image = image.permute(2, 0, 1)

    return image
