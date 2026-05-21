import logging

logger = logging.getLogger(__name__)
from typing import List

import torch
from typing import Union
from od3d_basic.cv.geometry.objects3d.objects3d import (
    PROJECT_MODALITIES,
    FEATS_DISTR,
    FEATS_ACTIVATION,
)

from typing import Optional
from od3d_basic.cv.geometry.objects3d.meshes.meshes import (
    Meshes,
    FACE_BLEND_TYPE,
    OD3D_Meshes_Deform,
    RASTERIZER,
)
from omegaconf import DictConfig


class Meshes_x_Gaussians(Meshes):
    def __init__(
        self,
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
        feat_dim=128,
        objects_count=0,
        feats_objects: Union[bool, torch.Tensor] = False,
        feats_objects_requires_param: bool = True,
        feat_clutter_requires_param: bool = True,
        verts_uvs: List[torch.Tensor] = None,
        feats_requires_grad=True,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        feats_activation=FEATS_ACTIVATION.NORM_DETACH,
        rgb: List[torch.Tensor] = None,
        verts_requires_grad=False,
        verts_requires_param=True,
        verts_coarse_count: int = 150,
        verts_coarse_prob_sigma: float = 0.01,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
        face_blend_type=FACE_BLEND_TYPE.SOFT_SIGMOID_NORMALIZED,
        rasterizer=RASTERIZER.NVDIFFRAST,
        face_blend_count=2,
        face_opacity=1.0,
        face_opacity_face_sdf_sigma=1e-4,
        face_opacity_face_sdf_gamma=1e-4,
        scale_3D_params: float = 1e2,
        instance_deform_net_config: DictConfig = None,
        gs_top_k=3,
        gs_scale=0.05,
        gs_opacity_requires_grad=False,
        gs_scale_requires_grad=False,
        gs_rotation_requires_grad=False,
    ):
        super().__init__(
            verts=verts,
            faces=faces,
            feat_dim=feat_dim,
            objects_count=objects_count,
            feats_objects=feats_objects,
            feats_objects_requires_param=feats_objects_requires_param,
            feat_clutter_requires_param=feat_clutter_requires_param,
            verts_uvs=verts_uvs,
            feats_requires_grad=feats_requires_grad,
            feat_clutter=feat_clutter,
            feats_distribution=feats_distribution,
            feats_activation=feats_activation,
            rgb=rgb,
            verts_requires_grad=verts_requires_grad,
            verts_requires_param=verts_requires_param,
            verts_coarse_count=verts_coarse_count,
            verts_coarse_prob_sigma=verts_coarse_prob_sigma,
            geodesic_prob_sigma=geodesic_prob_sigma,
            gaussian_splat_enabled=gaussian_splat_enabled,
            gaussian_splat_opacity=gaussian_splat_opacity,
            gaussian_splat_pts3d_size_rel_to_neighbor_dist=gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            pt3d_raster_perspective_correct=pt3d_raster_perspective_correct,
            device=device,
            dtype=dtype,
            rasterizer=rasterizer,
            face_blend_type=face_blend_type,
            face_blend_count=face_blend_count,
            face_opacity=face_opacity,
            face_opacity_face_sdf_sigma=face_opacity_face_sdf_sigma,
            face_opacity_face_sdf_gamma=face_opacity_face_sdf_gamma,
            scale_3D_params=scale_3D_params,
            instance_deform_net_config=instance_deform_net_config,
        )

        factory_kwargs = {"device": device, "dtype": dtype}

        if gs_opacity_requires_grad:
            self.verts_gs_opacity = torch.nn.Parameter(
                torch.ones((self.verts_count,)).to(**factory_kwargs),
                requires_grad=gs_opacity_requires_grad,
            )
        else:
            self.verts_gs_opacity = 1.0

        self.gs_top_k = gs_top_k
        if gs_scale_requires_grad:
            self.verts_gs_scale = torch.nn.Parameter(
                gs_scale * torch.ones((self.verts_count, 3)).to(**factory_kwargs),
                requires_grad=gs_scale_requires_grad,
            )
        else:
            self.verts_gs_scale = (
                gs_scale  #  * torch.ones((self.verts_count, 3)).to(**factory_kwargs)
            )

        gs_rotation_init = torch.zeros((self.verts_count, 4)).to(**factory_kwargs)
        gs_rotation_init[:, 0] = 1.0
        if gs_rotation_requires_grad:
            self.verts_gs_rotation = torch.nn.Parameter(
                gs_rotation_init,
                requires_grad=gs_rotation_requires_grad,
            )
        else:
            self.verts_gs_rotation = None

        # TODO add parameter opacity (init)
        # TODO add parameter pts3d size (init)
        # TODO add parameter pts3d rotation (init)

    def get_stacked_with_mesh_ids(self, tensor_packed, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        tensors_padded = []
        for mesh_id in mesh_ids:
            tensors_padded.append(
                self.get_tensor_verts_with_pad(
                    tensor_packed[
                        self.verts_counts_acc_from_0[
                            mesh_id
                        ] : self.verts_counts_acc_from_0[mesh_id + 1]
                    ],
                    mesh_id=mesh_id,
                ),
            )
        return torch.stack(tensors_padded, dim=0)

    def render_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
        instance_deform: OD3D_Meshes_Deform = None,
        **kwargs,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            add_clutter: bool, determines wether to add clutter features to the rendered features.
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        device = cams_tform4x4_obj.device
        dtype = cams_tform4x4_obj.dtype

        from od3d_basic.cv.render.gaussians_splats_v2 import (
            rasterize_gaussians,
            select_and_blend_gaussians,
        )

        pts3d = (
            self.get_stacked_with_mesh_ids(
                tensor_packed=self.verts,
                mesh_ids=objects_ids,
            )
            .clone()
            .to(device=device)
        )

        if isinstance(self.verts_gs_scale, torch.Tensor):
            pts3d_scale = (
                self.get_stacked_with_mesh_ids(
                    tensor_packed=self.verts_gs_scale,
                    mesh_ids=objects_ids,
                )
                .clone()
                .to(device=device)
            )
        else:
            pts3d_scale = self.verts_gs_scale
        if isinstance(self.verts_gs_rotation, torch.Tensor):
            pts3d_rotation = (
                self.get_stacked_with_mesh_ids(
                    tensor_packed=self.verts_gs_rotation,
                    mesh_ids=objects_ids,
                )
                .clone()
                .to(device=device)
            )
        else:
            pts3d_rotation = self.verts_gs_rotation

        if isinstance(self.verts_gs_opacity, torch.Tensor):
            pts3d_opacity = (
                self.get_stacked_with_mesh_ids(
                    tensor_packed=self.verts_gs_opacity,
                    mesh_ids=objects_ids,
                )
                .clone()
                .to(device=device)
            )
        else:
            pts3d_opacity = self.verts_gs_opacity

        if instance_deform is not None:
            pts3d += instance_deform.verts_deform

        pts3d_mask = (
            self.get_verts_stacked_mask_with_mesh_ids(mesh_ids=objects_ids)
            .clone()
            .to(device=device)
        )

        gs_id, px_to_gs_opacity, gs_depth, gs_normal = rasterize_gaussians(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_size=imgs_sizes,
            pts3d=pts3d,
            pts3d_mask=pts3d_mask,
            pts3d_size=pts3d_scale,
            pts3d_opacity=pts3d_opacity,
            pts3d_rotation=pts3d_rotation,
            topK=self.gs_top_k,
        )

        mods2d_rendered = {}
        for modality in modalities:
            if modality == PROJECT_MODALITIES.MASK:
                mod2d_rendered = px_to_gs_opacity.sum(dim=-1)[:, None]

            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
                or modality == PROJECT_MODALITIES.ONEHOT_COARSE
            ):
                if modality == PROJECT_MODALITIES.ONEHOT:
                    verts_one_hot_from_faces = self.get_labels_onehot(
                        smooth=False,
                        objects_ids=objects_ids,
                        add_other_objects=add_other_objects,
                        add_clutter=add_clutter,
                        device=device,
                    )[:, 0]

                elif modality == PROJECT_MODALITIES.ONEHOT_SMOOTH:
                    verts_one_hot_from_faces = self.get_labels_onehot(
                        smooth=True,
                        objects_ids=objects_ids,
                        add_other_objects=add_other_objects,
                        add_clutter=add_clutter,
                        device=device,
                    )[:, 0]

                elif modality == PROJECT_MODALITIES.ONEHOT_COARSE:
                    verts_one_hot_from_faces = self.get_labels_onehot(
                        coarse=True,
                        smooth=False,
                        objects_ids=objects_ids,
                        add_other_objects=add_other_objects,
                        add_clutter=add_clutter,
                        device=device,
                    )[:, 0]
                if add_clutter:
                    feat_bg = self.get_label_clutter(
                        add_other_objects=add_other_objects,
                        coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                        one_hot=True,
                        device=device,
                    ).to(dtype)[0] = self.get_label_clutter(
                        add_other_objects=add_other_objects,
                        coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                        one_hot=True,
                        device=device,
                    ).to(
                        dtype,
                    )[
                        0
                    ]
                else:
                    feat_bg = None

                mod2d_rendered = select_and_blend_gaussians(
                    gs_id,
                    px_to_gs_opacity,
                    verts_one_hot_from_faces,
                    feat_bg=feat_bg,
                )

            elif modality == PROJECT_MODALITIES.DEPTH:
                mod2d_rendered = select_and_blend_gaussians(
                    gs_id,
                    px_to_gs_opacity,
                    gs_depth,
                )

            elif modality == PROJECT_MODALITIES.MASK_VERTS_VSBL:
                B = px_to_gs_opacity.shape[0]
                verts_vsbl_mask = torch.zeros(
                    size=(B, self.verts_counts_max),
                    dtype=torch.bool,
                    device=device,
                )
                for b in range(B):
                    verts_vsbl_mask[
                        b,
                        gs_id[
                            b,
                            px_to_gs_opacity[b].mean(dim=0).mean(dim=1)[b] > 0,
                        ].unique(),
                    ] = 1
                mod2d_rendered = verts_vsbl_mask

                # B = fragments.pix_to_face.shape[0]
                # faces_ids = torch.cat(
                #     [
                #         self.get_faces_with_mesh_id(object_id)
                #         for object_id in objects_ids
                #     ],
                #     dim=0,
                # )
                # # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
                # verts_vsbl_mask = torch.zeros(
                #     size=(B, self.verts_counts_max),
                #     dtype=torch.bool,
                #     device=device,
                # )
                # for b in range(B):
                #     # logger.info(f'meshes_ids {meshes_ids}')
                #     faces_ids_vsbl = fragments.pix_to_face[b]
                #     faces_ids_vsbl = faces_ids_vsbl.unique()
                #     faces_ids_vsbl = faces_ids_vsbl[faces_ids_vsbl >= 0]
                #     verts_ids_vsbl = faces_ids[faces_ids_vsbl].unique()
                #     verts_vsbl_mask[b, verts_ids_vsbl] = 1
                # mod2d_rendered = verts_vsbl_mask

            elif modality == PROJECT_MODALITIES.FEATS:
                gs_feats = self.get_feats_stacked_with_mesh_ids(
                    mesh_ids=objects_ids,
                ).to(device)

                if add_clutter:
                    feat_bg = self.feat_clutter.clone()
                else:
                    feat_bg = None

                mod2d_rendered = select_and_blend_gaussians(
                    gs_id,
                    px_to_gs_opacity,
                    gs_feats,
                    feat_bg=feat_bg,
                )

            elif modality == PROJECT_MODALITIES.PT3D_NCDS:
                gs_feats = self.get_verts_ncds_stacked_with_mesh_ids(
                    mesh_ids=objects_ids,
                ).to(device)

                mod2d_rendered = select_and_blend_gaussians(
                    gs_id,
                    px_to_gs_opacity,
                    gs_feats,
                )

            else:
                raise ValueError(f"Unknown modality {modality}")

            mods2d_rendered[modality] = mod2d_rendered

        return mods2d_rendered
