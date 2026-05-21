import logging

logger = logging.getLogger(__name__)
import torch
from od3d_basic.cv.geometry.transform import (
    proj3d2d,
)
from od3d_basic.od3d_datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d_basic.cv.visual.crop import crop
from omegaconf import DictConfig
from od3d_basic.od3d_datasets.dtd import DTD
import torchvision
from od3d_basic.cv.transforms.transform import OD3D_Transform
from od3d_basic.cv.geometry.grid import get_pxl2d
from od3d_basic.cv.geometry.grid import get_pxl2d


class CenterZoom3D(OD3D_Transform):
    # resize types: fit to
    def __init__(
        self,
        H,
        W,
        scale=None,
        center_rel_shift_xy=[0.0, 0.0],
        apply_txtr=False,
        config: DictConfig = None,
        scale_with_mask=None,
        scale_with_dist=None,
        scale_with_pad=True,
        center_use_mask=False,
        scale_selection="shorter",
    ):
        super().__init__()
        self.center_rel_shift_xy = (
            torch.Tensor(center_rel_shift_xy)
            if center_rel_shift_xy is not None
            else None
        )
        self.H = H
        self.W = W
        self.center_use_mask = center_use_mask
        self.scale = scale
        self.scale_with_mask = scale_with_mask
        self.scale_with_dist = scale_with_dist
        self.scale_with_pad = scale_with_pad
        self.scale_selection = (
            scale_selection  # 'separate' # 'shorter' 'larger' 'separate'
        )
        self.apply_txtr = apply_txtr
        if self.apply_txtr:
            self.dtd = DTD.create_from_config(
                config=config,
                transform=torchvision.transforms.Compose([]),
            )
        self.mode_rgb = "bilinear"  # "bilinear" "nearest_v2"
        self.mode_depth = "nearest_v2"  # "bilinear" "nearest_v2"
        self.mode_mask = "nearest_v2"  # "bilinear" "nearest_v2""
        self.mode_pxl_cat_id = "nearest_v2"

    def __call__(self, frame: OD3D_Frame):
        # logger.info(f"Frame name {self.name}")
        # _, _, _, _ = frame.size, frame.cam_intr4x4, frame.cam_tform4x4_obj, frame.cam_proj4x4_obj

        if frame.get_cam_tform4x4_obj()[2, 3] <= 0.0:
            logger.warning(f"dist <= 0")

        if self.center_use_mask and (frame.get_mask() > 0.5).sum() > 0:
            mask = frame.get_mask() > 0.5
            mask_pxl2d = get_pxl2d(
                H=mask.shape[1],
                W=mask.shape[2],
                dtype=float,
                device=mask.device,
            )
            mask_pxl2d = mask_pxl2d[mask[0]]
            # this automatic scales to fit the cropped image
            center2d = (
                mask_pxl2d.min(dim=0).values
                + (mask_pxl2d.max(dim=0).values - mask_pxl2d.min(dim=0).values) / 2.0
            ).to(device=mask.device)
        else:
            if self.center_rel_shift_xy is not None and frame.get_cam_tform4x4_obj()[2, 3] > 0.0:
                center2d = proj3d2d(
                    torch.Tensor([0.0, 0.0, 0.0]),
                    proj4x4=frame.cam_proj4x4_obj,
                )
                
                if (
                    #(self.center_rel_shift_xy[0] > 0.
                    #or self.center_rel_shift_xy[1] > 0.)
                    #and 
                    # note: center can also be far outside without rel shift. 
                    (center2d[0] < -frame.W
                    or center2d[1] < -frame.H
                    or center2d[0] > 2 * frame.W
                    or center2d[1] > 2 * frame.H)
                ):
                    logger.warning(
                        f"center {center2d[0].item():.2f}, {center2d[1].item():.2f} far outside of frame {frame.W}, {frame.H}. Setting center to image center.",
                    )
                    center2d = (frame.size.flip(dims=[0])-1) / 2
                if center2d.isnan().any():
                    center2d = (frame.size.flip(dims=[0])-1) / 2
            else:
                center2d = (frame.size.flip(dims=[0]) - 1) / 2

        if self.scale_with_dist is not None:
            # note: this usage should become deprecated in the future.
            from od3d_basic.od3d_datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL

            dist = self.scale_with_dist
            if (
                frame.category is not None
                and frame.category in PASCAL3D_SCALE_NORMALIZE_TO_REAL.keys()
            ):
                dist *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[frame.category]
            scale = frame.get_cam_tform4x4_obj()[2, 3] / dist

        elif self.scale_with_mask is not None and frame.get_mask() is not None:
            # note: this usage should become deprecated in the future.
            if self.scale is not None:
                logger.warning(
                    "For CenterZoom3D `scale` and `scale_with_mask` are not None. Only using `scale_with_mask`.",
                )

            mask = frame.get_mask() > 0.5
            if mask.sum() > 0.0:
                mask_pxl2d = get_pxl2d(
                    H=mask.shape[1],
                    W=mask.shape[2],
                    dtype=float,
                    device=mask.device,
                )
                mask_pxl2d = mask_pxl2d[mask[0]]
                # this automatic scales to fit the cropped image
                mask_H = int(
                    max(
                        mask_pxl2d[:, 1].max() - center2d[1],
                        center2d[1] - mask_pxl2d[:, 1].min(),
                    )
                    * 2,
                )
                mask_H = max(mask_H, 1)

                mask_W = int(
                    max(
                        mask_pxl2d[:, 0].max() - center2d[0],
                        center2d[0] - mask_pxl2d[:, 0].min(),
                    )
                    * 2,
                )
                mask_W = max(mask_W, 1)

            else:
                logger.warning(
                    "For CenterZoom3D using `scale_with_mask` despite mask has only zeros. Setting mask width and height to image width and height.",
                )
                mask_H = self.H
                mask_W = self.W

            if self.scale_selection == "shorter":
                scale = torch.Tensor(
                    [
                        min(
                            (self.scale_with_mask * self.H) / mask_H,
                            (self.W * self.scale_with_mask) / mask_W,
                        ),
                    ],
                )
            elif self.scale_selection == "larger":
                scale = torch.Tensor(
                    [
                        max(
                            (self.scale_with_mask * self.H) / mask_H,
                            (self.W * self.scale_with_mask) / mask_W,
                        ),
                    ],
                )
            elif self.scale_selection == "separate":
                scale = torch.Tensor(
                    [
                        (self.W * self.scale_with_mask) / mask_W,
                        (self.scale_with_mask * self.H) / mask_H,
                    ],
                )
            else:
                msg = f"Unexepcted scale selection type: {self.scale_selection}."
                raise Exception(msg)
        else:
            if self.scale_with_mask is not None:
                logger.warning(
                    "For CenterZoom3D `scale_with_mask` is not None, but frame.mask is None. Ignoring `scale_with_mask`",
                )
            # this automatic scales to fit the cropped image
            centered_frame_H = int(max(frame.H - center2d[1], center2d[1]) * 2)
            centered_frame_W = int(max(frame.W - center2d[0], center2d[0]) * 2)
            if self.scale_selection == "shorter":
                scale = torch.Tensor(
                    [min(self.H / centered_frame_H, self.W / centered_frame_W)],
                )
            elif self.scale_selection == "larger":
                scale = torch.Tensor(
                    [max(self.H / centered_frame_H, self.W / centered_frame_W)],
                )
            elif self.scale_selection == "separate":
                scale = torch.Tensor(
                    [self.W / centered_frame_W, self.H / centered_frame_H],
                )
            else:
                msg = f"Unexepcted scale selection type: {self.scale_selection}."
                raise Exception(msg)

        if self.scale is not None:
            # scale = frame.get_cam_tform4x4_obj[2, 3] / self.dist
            scale *= self.scale

        # logger.info(f'scale = {scale}')
        if scale.mean() < 0.01:
            logger.warning(f"Scale is < 0.01. Setting scale to 0.99")
            scale[:] = 0.99

        if scale.mean() > 100.0:
            logger.warning(f"Scale is > 100. Setting scale to 1.01")
            scale[:] = 1.01

        center2d_shifted = center2d.clone()
        if self.center_rel_shift_xy is not None:
            center2d_shifted[0] += frame.W * self.center_rel_shift_xy[0]
            center2d_shifted[1] += frame.H * self.center_rel_shift_xy[1]

        # logger.info(f"scale {scale}, center2d {center2d_shifted}, W in {frame.W} H in {frame.H}")

        frame.rgb_mask, _ = crop(
            frame.get_rgb_mask(),
            center=center2d_shifted,
            H_out=self.H,
            W_out=self.W,
            scale=scale,
            ctx=None,
            mode=self.mode_mask,
        )

        if OD3D_FRAME_MODALITIES.MASK in frame.modalities:
            frame.mask, _ = crop(
                img=frame.get_mask(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_mask,
            )

        if OD3D_FRAME_MODALITIES.PXL_CAT_ID in frame.modalities:
            frame.pxl_cat_id, _ = crop(
                img=frame.get_pxl_cat_id(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_pxl_cat_id,
            )

        if OD3D_FRAME_MODALITIES.DEPTH in frame.modalities:
            frame.depth, _ = crop(
                img=frame.get_depth(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_depth,
            )

        if OD3D_FRAME_MODALITIES.DEPTH_MASK in frame.modalities:
            depth_mask = frame.get_depth_mask() #hotfix: for omni6dpose bbox crop have shape (1, 1, H, W)
            depth_mask = depth_mask.reshape(-1, depth_mask.shape[-2], depth_mask.shape[-1])
            frame.depth_mask, _ = crop(
                img=depth_mask,
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_mask,
            )

        # mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale, ctx=self.txtr)
        if self.apply_txtr:
            frame.rgb, cam_crop_tform_cam = crop(
                img=frame.get_rgb(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=self.dtd.get_random_item().rgb,
                mode=self.mode_rgb,
            )
        else:
            frame.rgb, cam_crop_tform_cam = crop(
                img=frame.get_rgb(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_rgb,
            )

        if OD3D_FRAME_MODALITIES.RGB_ORIG in frame.modalities:
            frame.rgb_orig, cam_crop_tform_cam = crop(
                img=frame.get_rgb_orig(),
                center=center2d_shifted,
                H_out=self.H,
                W_out=self.W,
                scale=scale,
                ctx=None,
                mode=self.mode_rgb,
            )

        frame.size[0:1] = self.H
        frame.size[1:2] = self.W

        frame.cam_intr4x4 = torch.bmm(
            cam_crop_tform_cam[None,],
            frame.get_cam_intr4x4()[None,],
        )[0]
        frame.get_cam_tform4x4_obj()

        if self.scale_with_dist is not None:
            # note: this usage should become deprecated in the future.
            frame.cam_intr4x4[:2, :2] *= 1.0 / scale
            frame.cam_tform4x4_obj[2, 3] *= 1.0 / scale

        if OD3D_FRAME_MODALITIES.BBOX in frame.modalities:
            from od3d_basic.cv.geometry.transform import transf2d_bbox
            frame.bbox = transf2d_bbox(bbox=frame.get_bbox().clone(), transf3x3=cam_crop_tform_cam)

            # # x_min, y_min, x_max, y_max
            # frame.bbox = (frame.get_bbox().reshape(2, 2) * scale[None,]).flatten()
            # frame.bbox[[0, 2]] = frame.bbox[[0, 2]] + cam_crop_tform_cam[0, 2]
            # frame.bbox[[1, 3]] = frame.bbox[[1, 3]] + cam_crop_tform_cam[1, 2]

        if OD3D_FRAME_MODALITIES.BBOXS in frame.modalities:
            # x_min, y_min, x_max, y_max
            from od3d_basic.cv.geometry.transform import transf2d_bbox
            frame.bboxs = transf2d_bbox(bbox=frame.get_bboxs().clone(), transf3x3=cam_crop_tform_cam)
            #
            # frame.bboxs = (
            #     frame.get_bboxs().reshape(-1, 2, 2) * scale[None, None]
            # ).flatten(1)
            # frame.bboxs[:, [0, 2]] = (
            #     frame.bboxs[:, [0, 2]] + cam_crop_tform_cam[0, 2][None,]
            # )
            # frame.bboxs[:, [1, 3]] = (
            #     frame.bboxs[:, [1, 3]] + cam_crop_tform_cam[1, 2][None,]
            # )

        if OD3D_FRAME_MODALITIES.KPTS2D_ANNOT in frame.modalities:
            frame.kpts2d_annot = frame.get_kpts2d_annot() * scale[None,]
            frame.kpts2d_annot = frame.kpts2d_annot + cam_crop_tform_cam[:2, 2]

        # assumption: depth of all image points is the same (which of course does only approximately holds)
        # frame.cam_tform4x4_obj[:2, 3] = 0.
        # frame.cam_intr4x4[0, 2] = self.W / 2.
        # frame.cam_intr4x4[1, 2] = self.H / 2.
        #from od3d_basic.cv.visual.show import show_imgs
        #show_imgs([frame.depth == 0, frame.depth < 0.1])

        if hasattr(frame, 'pair') and frame.pair is not None:
            frame.pair = self(frame.pair)

        return frame

        """
        if frame.fpath_shapenemo is not None:
            shapenemo_mesh = Mesh(fpath_mesh=frame.fpath_shapenemo)
            frame.shapenemo_vts3d = shapenemo_mesh.verts # load_mesh_vertices(fpath_mesh=self.fpath_shapenemo, device=self.device)
            frame.shapenemo_mask, frame.shapenemo_depth, frame.vts2d, frame.vts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=self.fpath_shapenemo, pts3d=self.shapenemo_vts3d)

        frame.mask, frame.depth, frame.kpts2d, frame.kpts3d_vsbl = frame.calc_mesh_proj(fpath_mesh=frame.fpath_mesh, pts3d=frame.kpts3d)
        """
