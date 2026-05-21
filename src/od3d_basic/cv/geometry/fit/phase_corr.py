from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import get_affine_matrix2d
from kornia.geometry.transform import warp_affine

logger = getLogger(__name__)


def get_affine_from_imgs(img_a, img_b):
    H, W = img_a.shape[-2:]
    device = img_a.device
    h = logpolar_filter((H, W)).to(device=device)  # highpass [H,W]

    img_a_fft = torch.view_as_real(torch.fft.fft2(img_a))
    img_a_fft = torch.sqrt(img_a_fft[..., 0] ** 2 + img_a_fft[..., 1] ** 2 + 1e-15)
    img_a_fft = torch.fft.fftshift(img_a_fft, dim=(-2, -1))
    img_a_fft *= h[None,]

    img_b_fft = torch.view_as_real(torch.fft.fft2(img_b))
    img_b_fft = torch.sqrt(img_b_fft[..., 0] ** 2 + img_b_fft[..., 1] ** 2 + 1e-15)
    img_b_fft = torch.fft.fftshift(img_b_fft, dim=(-2, -1))
    img_b_fft *= h[None,]

    img_a_fft_polar, img_a_fft_polar_logbase = polar_transformer(
        img_a_fft[..., None],
        out_size=(H, W),
        device=device,
    )
    img_a_fft_polar = img_a_fft_polar.permute(3, 0, 1, 2)[0]

    img_b_fft_polar, img_b_fft_polar_logbase = polar_transformer(
        img_b_fft[..., None],
        out_size=(H, W),
        device=device,
    )
    img_b_fft_polar = img_b_fft_polar.permute(3, 0, 1, 2)[0]

    index_corr_max_x, index_corr_max_y = get_corr_amax(img_a_fft_polar, img_b_fft_polar)

    angle_ori = -index_corr_max_y / (H) * 180
    angle = angle_ori.clone()
    # angle[angle_ori < 0] += 360
    # angle[angle_ori >= 90] -= 90.
    # angle[angle_ori < 90] += 90.

    scale = torch.pow(img_b_fft_polar_logbase, -index_corr_max_x.float())

    scale = torch.Tensor([scale, scale]).to(device)[None,].repeat(2, 1)
    angle = torch.Tensor([angle, angle + 180]).to(device)
    transl = torch.Tensor([0, 0]).to(device)[None,].repeat(2, 1)
    center = torch.Tensor([W / 2, H / 2]).to(device)[None,].repeat(2, 1)
    # center = torch.Tensor([0, 0]).to(device)[None,]
    M = get_affine_matrix2d(
        translations=transl,
        angle=angle,
        scale=scale,
        center=center,
    )
    imgs_a_rot_scaled = warp_affine(
        img_a[None,].repeat(2, 1, 1, 1),
        M=M[:, :2, :3],
        dsize=(H, W),
    )

    transl1_x, transl1_y, corr1 = get_corr_amax(
        imgs_a_rot_scaled[0],
        img_b,
        return_corr=True,
    )
    transl2_x, transl2_y, corr2 = get_corr_amax(
        imgs_a_rot_scaled[1],
        img_b,
        return_corr=True,
    )
    if corr1 > corr2:
        transl_x, transl_y = transl1_x, transl1_y
    else:
        transl_x, transl_y = transl2_x, transl2_y
        angle = angle + 180.0

    logger.info(f"angle {angle}, scale {scale}")
    logger.info(f"transl_x {transl_x}, transl_y {transl_y}")

    transl = torch.Tensor([transl_x, transl_y]).to(device)[None,]

    M = get_affine_matrix2d(
        translations=transl,
        angle=angle,
        scale=scale,
        center=center,
    )
    return M[0]


def get_corr_amax(img_a, img_b, return_corr=False):
    device = img_a.device
    H, W = img_a.shape[-2:]
    eps = 1e-15

    G_a = torch.view_as_real(torch.fft.fft2(img_a))  # , onesided=False)
    G_b = torch.view_as_real(torch.fft.fft2(img_b))  # , onesided=False)

    # torch.view_as_real(G_a
    real_a = G_a[:, :, :, 0]
    real_b = G_b[:, :, :, 0]
    imag_a = G_a[:, :, :, 1]
    imag_b = G_b[:, :, :, 1]

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(G_a.shape[0], G_a.shape[1], G_a.shape[2], 2).to(device)
    R[:, :, :, 0] = real_a * real_b + imag_a * imag_b
    R[:, :, :, 1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a**2 + imag_a**2 + eps) * torch.sqrt(
        real_b**2 + imag_b**2 + eps,
    )
    R[:, :, :, 0] = R[:, :, :, 0].clone() / (r0 + eps).to(device)
    R[:, :, :, 1] = R[:, :, :, 1].clone() / (r0 + eps).to(device)

    r = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(R)))
    r_real = r[:, :, :, 0]
    r_imag = r[:, :, :, 1]

    r = torch.sqrt(r_real**2 + r_imag**2 + eps)

    r = torch.fft.ifftshift(r, dim=(-2, -1))
    r_sum = r.sum(dim=-3)
    index_corr_max = r_sum.flatten(-2).argmax(dim=-1)  # .shape # shape
    index_corr_max_x = index_corr_max % H - W // 2
    index_corr_max_y = index_corr_max // H - H // 2
    if return_corr:
        return index_corr_max_x, index_corr_max_y, r_sum.flatten(-2)[index_corr_max]
    else:
        return index_corr_max_x, index_corr_max_y


def logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[0])[:, np.newaxis]
    xx = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy**2 + xx**2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    filt = torch.from_numpy(filt)
    return filt


def polar_transformer(U, out_size, device, log=True, radius_factor=0.707):
    """Polar Transformer Layer

    Based on https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py.
    _repeat(), _interpolate() are exactly the same;
    the polar transform implementation is in _transform()

    Args:
        U, theta, out_size, name: same as spatial_transformer.py
        log (bool): log-polar if True; else linear polar
        radius_factor (float): 2maxR / Width
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(n_repeats)
        rep.unsqueeze(0)
        x = torch.reshape(x, (-1, 1))
        x = x * rep
        return torch.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):  # im [B,H,W,C]
        # constants
        x = x.to(device)
        y = y.to(device)
        num_batch = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]
        height_f = height
        width_f = width

        x = x.double()
        y = y.double()
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([])
        max_y = im.shape[1] - 1
        max_x = im.shape[2] - 1

        # do sampling
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width * height

        # base = _repeat(torch.arange(0, num_batch, dtype=int)*dim1, out_height*out_width)
        base = torch.tile(
            torch.arange(0, num_batch, dtype=int, device=device).unsqueeze(1) * dim1,
            (1, out_height * out_width),
        ).reshape(-1)
        base = base.long()
        # base = base.to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])

        im_flat = im_flat.clone().float()  # .to(device)

        Ia = im_flat.gather(0, idx_a.unsqueeze(1).type(torch.int64))
        Ib = im_flat.gather(0, idx_b.unsqueeze(1).type(torch.int64))
        Ic = im_flat.gather(0, idx_c.unsqueeze(1).type(torch.int64))
        Id = im_flat.gather(0, idx_d.unsqueeze(1).type(torch.int64))

        # print(im_flat.shape, Id.shape, idx_d.shape)

        # Ia = im_flat[idx_a].to(device)
        # Ib = im_flat[idx_b].to(device)
        # Ic = im_flat[idx_c].to(device)
        # Id = im_flat[idx_d].to(device)

        # and finally calculate interpolated values
        x0_f = x0.double()
        x1_f = x1.double()
        y0_f = y0.double()
        y1_f = y1.double()
        # print(x0_f.shape, x0.shape, x.shape)

        # print(((x1_f-x) * (y1_f-y)).shape)
        # print("-------------")
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        # output = Ia + Ib + Ic + Id
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width):
        x_t = (
            torch.linspace(0.0, 1.0 * width - 1, width, device=device)
            .unsqueeze(0)
            .repeat(height, 1)
        )
        y_t = (
            torch.linspace(0.0, 1.0, height, device=device)
            .unsqueeze(1)
            .repeat(1, width)
        )
        # x_t = torch.ones([height, 1]) * torch.linspace(0.0, 1.0 * width-1, width).unsqueeze(1).permute(1, 0)
        # y_t = torch.linspace(0.0, 1.0, height).unsqueeze(1) * torch.ones([1, width])

        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))
        grid = torch.cat((x_t_flat, y_t_flat), 0)

        return grid

    def _transform(input_dim, out_size):
        # radius_factor = torch.sqrt(torch.tensor(2.))/2.
        num_batch = input_dim.shape[0]  # input [B,H,W,C]
        num_channels = input_dim.shape[3]

        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)  # (2, WxH)
        grid = grid.unsqueeze(0)
        grid = torch.reshape(grid, [-1])
        grid = grid.repeat(num_batch)
        grid = torch.reshape(grid, [num_batch, 2, -1])  # (B,2,WxH)

        ## here we do the polar/log-polar transform
        W = torch.tensor(input_dim.shape[1], dtype=torch.double, device=device)
        # W = input_dim.shape[1].float()
        maxR = W * radius_factor

        # if radius is from 1 to W/2; log R is from 0 to log(W/2)
        # we map the -1 to +1 grid to log R
        # then remap to 0 to 1
        EXCESS_CONST = 1.1

        logbase = torch.exp(
            torch.log(W * EXCESS_CONST / 2) / W,
        )  # 10. ** (torch.log10(maxR) / W)
        # torch.exp(torch.log(W*EXCESS_CONST/2) / W) #
        # get radius in pix
        if log:
            # min=1, max=maxR
            r_s = torch.pow(logbase, grid[:, 0, :])
        else:
            # min=1, max=maxR
            r_s = 1 + (grid[:, 0, :] + 1) / 2 * (maxR - 1)

        # y is from -1 to 1; theta is from 0 to 2pi
        theta = np.linspace(0.0, np.pi, input_dim.shape[1], endpoint=False) * -1.0
        # t_s = torch.from_numpy(theta).unsqueeze(1) * torch.ones([1, out_width])
        t_s = torch.from_numpy(theta).to(device).unsqueeze(1).repeat(1, out_width)
        t_s = torch.reshape(t_s, (1, -1))

        # use + theta[:, 0] to deal with origin
        x_s = r_s * torch.cos(t_s) + (W / 2)
        y_s = r_s * torch.sin(t_s) + (W / 2)

        x_s_flat = torch.reshape(x_s, [-1])
        y_s_flat = torch.reshape(y_s, [-1])

        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        output = torch.reshape(
            input_transformed,
            [num_batch, out_height, out_width, num_channels],
        )  # .to(device)
        return output, logbase

    output, logbase = _transform(U, out_size)
    return [output, logbase]
