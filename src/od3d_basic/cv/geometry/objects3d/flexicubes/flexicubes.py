import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import List

import torch
from typing import Union
from od3d_basic.cv.geometry.objects3d.objects3d import (
    PROJECT_MODALITIES,
    FEATS_DISTR,
    FEATS_ACTIVATION,
)

from typing import Optional
from od3d_basic.cv.geometry.objects3d.meshes_x_gaussians import Meshes_x_Gaussians
from od3d_basic.cv.geometry.objects3d.meshes.meshes import (
    Meshes,
    FACE_BLEND_TYPE,
    OD3D_Meshes_Deform,
    RASTERIZER,
)
from omegaconf import DictConfig


class Flexicubes(Meshes):
    def __init__(
        self,
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
        feat_dim=128,
        objects_count=0,
        feats_objects: Union[bool, torch.Tensor] = False,
        verts_uvs: List[torch.Tensor] = None,
        verts_coarse_count: int = 150,
        verts_coarse_prob_sigma: float = 0.01,
        feats_requires_grad=True,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        feats_activation=FEATS_ACTIVATION.NORM_DETACH,
        rgb: List[torch.Tensor] = None,
        verts_requires_grad=False,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
        rasterizer=RASTERIZER.NVDIFFRAST,
        face_blend_type=FACE_BLEND_TYPE.SOFT_SIGMOID_NORMALIZED,
        face_blend_count=2,
        face_opacity=1.0,
        face_opacity_face_sdf_sigma=1e-4,
        face_opacity_face_sdf_gamma=1e-4,
        instance_deform_net_config: DictConfig = None,
        tet_res=16,
        **kwargs,
    ):
        super().__init__(
            verts=verts,
            faces=faces,
            feat_dim=feat_dim,
            objects_count=objects_count,
            feats_objects=None,
            verts_uvs=verts_uvs,
            feats_requires_grad=feats_requires_grad,
            feats_objects_requires_param=False,
            feat_clutter=feat_clutter,
            feats_distribution=feats_distribution,
            feats_activation=feats_activation,
            rgb=rgb,
            verts_requires_grad=verts_requires_grad,
            verts_requires_param=False,
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
            instance_deform_net_config=instance_deform_net_config,
            **kwargs,
        )

        if dtype is None:
            dtype = torch.float

        self.init_radius = 0.9
        # self.mesh_update_jitter_scale = 0.05

        # https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb
        self.fc_tet_res = tet_res
        import kaolin as kal

        self.fc = kal.non_commercial.FlexiCubes(device)
        self.fc_x_nx3, self.fc_cube_fx8 = self.fc.construct_voxel_grid(self.fc_tet_res)
        self.fc_range = 2.2
        self.fc_x_nx3 *= (
            self.fc_range
        )  # scale up the grid so that it's larger than the target object
        self.fc_x_nx3 = self.fc_x_nx3.to(device, dtype)
        self.fc_cube_fx8 = self.fc_cube_fx8.to(device)
        # range: -1, 1

        #  Retrieve all the edges of the voxel grid; these edges will be utilized to
        #  compute the regularization loss in subsequent steps of the process.
        self.fc_all_edges = (
            self.fc_cube_fx8[:, self.fc.cube_edges].reshape(-1, 2).to(device)
        )
        self.fc_grid_edges = torch.unique(self.fc_all_edges, dim=0).to(device)

        self.fc_sdfs = torch.nn.ParameterList()
        self.dual_vertex_center_dev_loss = []
        self.fc_deforms = torch.nn.ParameterList()
        self.fc_weights = torch.nn.ParameterList()

        self.feat_coordmlps = torch.nn.ModuleList()

        feat_coord_mlp_cfg = DictConfig(
            {
                "num_layers": 5,
                "hidden_dim": 256,
                "out_dim": feat_dim,
                "dropout": 0,
                "activation": None,
                "symmetrize": False,
            },
        )

        # note: import on-the-fly to avoid circular import
        from od3d_basic.models.heads.coordmlp import CoordMLP

        for m in range(self.meshes_count):
            # grid_scale = self.get_ranges()[m]
            # embedder_scalar = 2 * np.pi / grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9

            self.fc_deforms.append(
                torch.nn.Parameter(torch.zeros_like(self.fc_x_nx3), requires_grad=True),
            )

            _sdf = self.fc_x_nx3.detach().norm(dim=-1, keepdim=False) - self.init_radius
            # _sdf = torch.rand_like(self.x_nx3[:, 0]) - 0.1  # randomly initialize SDF
            self.fc_sdfs.append(
                torch.nn.Parameter(_sdf.clone().detach(), requires_grad=True),
            )

            # set per-cube learnable weights to zeros
            _weight = torch.zeros(
                (self.fc_cube_fx8.shape[0], 21),
                dtype=dtype,
                device=device,
            )
            self.fc_weights.append(
                torch.nn.Parameter(_weight.clone().detach(), requires_grad=True),
            )

            self.feat_coordmlps.append(
                CoordMLP(
                    in_dims=[0],
                    in_upsample_scales=[],
                    config=feat_coord_mlp_cfg,
                    n_harmonic_functions=8,
                    embed_concat_pts=True,
                ).to(device, dtype),
            )

        self.fc_weights = self.fc_weights.to(device, dtype)
        self.fc_sdfs = self.fc_sdfs.to(device, dtype)
        self.fc_deforms = self.fc_deforms.to(device, dtype)
        self.feat_coordmlps = self.feat_coordmlps.to(device, dtype)

        if not verts_requires_grad:
            for param in self.fc_sdfs.parameters():
                param.requires_grad = False
            for param in self.fc_weights.parameters():
                param.requires_grad = False
            for param in self.fc_deforms.parameters():
                param.requires_grad = False
        if not feats_requires_grad:
            for param in self.feat_coordmlps.parameters():
                param.requires_grad = False

        # from pathlib import Path
        # tets = np.load(Path(__file__).parent.resolve().joinpath('./{}_tets.npz'.format(str(tet_res))))

        # self.tets_scale = self.init_radius * 2.2
        # self.tets_verts = (torch.tensor(tets['vertices'], dtype=dtype, device=device)) * self.tets_scale  # verts original scale (-0.5, 0.5)
        # self.tets_faces = torch.tensor(tets['indices'], dtype=torch.long, device=device)
        # self.generate_edges()

        self.update_dmtet(device=device, dtype=dtype, require_grad=False)

    def set_verts_requires_grad(self, verts_requires_grad):
        self.verts_requires_grad = verts_requires_grad
        if not verts_requires_grad:
            for param in self.fc_sdfs.parameters():
                param.requires_grad = verts_requires_grad
            for param in self.fc_weights.parameters():
                param.requires_grad = verts_requires_grad
            for param in self.fc_deforms.parameters():
                param.requires_grad = verts_requires_grad

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        if "device" in kwargs.keys():
            import kaolin as kal

            self.fc = kal.non_commercial.FlexiCubes(kwargs["device"])

        self.fc_x_nx3 = self.fc_x_nx3.to(*args, **kwargs)
        self.fc_cube_fx8 = self.fc_cube_fx8.to(*args, **kwargs)

        self.fc_all_edges = self.fc_all_edges.to(*args, **kwargs)
        self.fc_grid_edges = self.fc_grid_edges.to(*args, **kwargs)

        # self.fc_sdfs.to(*args, **kwargs)
        # self.fc_deforms.to(*args, **kwargs)
        # self.fc_weights.to(*args, **kwargs)
        # self.feat_coordmlps.to(*args, **kwargs)

        # self.marching_tets.to(*args, **kwargs)
        # self.tets_verts = self.tets_verts.to(*args, **kwargs)
        # self.tets_faces = self.tets_faces.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        import kaolin as kal

        self.fc = kal.non_commercial.FlexiCubes(device="cuda")

        self.fc_x_nx3 = self.fc_x_nx3.cuda(*args, **kwargs)
        self.fc_cube_fx8 = self.fc_cube_fx8.cuda(*args, **kwargs)

        self.fc_all_edges = self.fc_all_edges.cuda(*args, **kwargs)
        self.fc_grid_edges = self.fc_grid_edges.cuda(*args, **kwargs)

        # self.fc_sdfs.cuda(*args, **kwargs)
        # self.fc_deforms.cuda(*args, **kwargs)
        # self.fc_weights.cuda(*args, **kwargs)
        # self.feat_coordmlps.cuda(*args, **kwargs)

        # self.marching_tets.cuda(*args, **kwargs)
        # self.tets_verts = self.tets_verts.cuda(*args, **kwargs)
        # self.tets_faces = self.tets_faces.cuda(*args, **kwargs)

    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        self.update_dmtet(require_grad=False)

    def update_dmtet(self, device=None, dtype=None, require_grad=None):
        if require_grad is None:
            require_grad = self.verts_requires_grad

        # kaolin.non_commercial.FlexiCubes
        # https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html#kaolin.non_commercial.FlexiCubes
        verts = []
        faces = []
        feats = []
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.dual_vertex_center_dev_loss = []
        for m in range(self.meshes_count):
            if require_grad:
                # extract and render FlexiCubes mesh
                _grid_verts = self.fc_x_nx3 + (self.fc_range - 1e-8) / (
                    self.fc_tet_res * 2
                ) * torch.tanh(self.fc_deforms[m])
                _verts, _faces, L_dev = self.fc(
                    _grid_verts,
                    self.fc_sdfs[m],
                    self.fc_cube_fx8,
                    self.fc_tet_res,
                    beta=self.fc_weights[m][:, :12],
                    alpha=self.fc_weights[m][:, 12:20],
                    gamma_f=self.fc_weights[m][:, 20],
                    training=True,
                )
                _feats = self.get_feats(pts=_verts.detach(), object_id=m)
            else:
                with torch.no_grad():
                    # extract and render FlexiCubes mesh
                    _grid_verts = self.fc_x_nx3 + (self.fc_range - 1e-8) / (
                        self.fc_tet_res * 2
                    ) * torch.tanh(self.fc_deforms[m])
                    _verts, _faces, L_dev = self.fc(
                        _grid_verts,
                        self.fc_sdfs[m],
                        self.fc_cube_fx8,
                        self.fc_tet_res,
                        beta=self.fc_weights[m][:, :12],
                        alpha=self.fc_weights[m][:, 12:20],
                        gamma_f=self.fc_weights[m][:, 20],
                        training=False,
                    )
                    _feats = self.get_feats(pts=_verts.detach(), object_id=m)

            self.dual_vertex_center_dev_loss.append(L_dev)
            verts.append(_verts)
            faces.append(_faces)
            feats.append(_feats)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.dual_vertex_center_dev_loss = torch.stack(self.dual_vertex_center_dev_loss)

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.verts_counts_max = max(self.verts_counts)

        self.meshes_count = len(verts)
        self.verts = torch.cat([_verts for _verts in verts], dim=0).to(**factory_kwargs)
        self.feats_objects = torch.cat([_feats for _feats in feats], dim=0).to(
            **factory_kwargs,
        )

        self.faces = torch.cat([_faces for _faces in faces], dim=0).to(device=device)

        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [
            sum(self.verts_counts[: i + 1]) for i in range(self.meshes_count)
        ]
        self.faces_counts_acc_from_0 = [0] + [
            sum(self.faces_counts[: i + 1]) for i in range(self.meshes_count)
        ]

        self.verts_count = self.verts_counts_acc_from_0[-1]
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(
            size=[len(self), self.verts_counts_max],
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i] :] = False

        import matplotlib.pyplot as plt

        color_ = plt.get_cmap("tab20", len(self))
        self.feats_rgb_object_id = []
        for i in range(len(self)):
            self.feats_rgb_object_id.extend([color_(i)] * self.verts_counts[i])

        self.update_verts_coarse()

    def get_uniform_jittered_tets_verts(self, object_id):
        pts = self.tets_verts.clone()  #  self.get_verts_with_mesh_id(object_id)
        if self.mesh_update_jitter_scale > 0:
            jitter = (
                (torch.rand(3, device=pts.device) - 0.5)
                * self.tets_scale
                * self.mesh_update_jitter_scale
            )
            pts = pts + jitter
        return pts

    def get_rand_jittered_mesh_verts(self, object_id):
        pts = self.get_verts_with_mesh_id(object_id, clone=True).detach()
        if self.mesh_update_jitter_scale > 0:
            jitter = (
                (torch.rand_like(pts, device=pts.device) - 0.5)
                * self.tets_scale
                * self.mesh_update_jitter_scale
            )
            pts = pts + jitter
        return pts

    def get_sdf(self, pts, object_id):
        """
        Args:
            pts (torch.Tensor): BxNx3
        Returns:
            sdf (torch.Tensor): BxN
        """

        sdf_init = pts.detach().norm(dim=-1, keepdim=True) - self.init_radius
        from od3d_basic.data.batch_datatypes import OD3D_ModelData

        sdf_delta = self.sdf_coordmlps[object_id](
            OD3D_ModelData(pts3d=pts[None,]),
        ).feat[0]
        sdf_vals = sdf_init + sdf_delta
        return sdf_vals

    def get_feats(self, pts, object_id):
        from od3d_basic.data.batch_datatypes import OD3D_ModelData

        feats = self.feat_coordmlps[object_id](OD3D_ModelData(pts3d=pts[None,])).feat[0]
        return feats

    def get_sdf_gradient(self, object_id):
        num_samples = 5000
        sample_points = (
            torch.rand(num_samples, 3, device=self.verts.device) - 0.5
        ) * self.tets_scale

        mesh_verts = self.get_rand_jittered_mesh_verts(object_id=object_id)

        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        y = self.get_sdf(pts=sample_points, object_id=object_id)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients

    def get_geo_sdf_reg_loss(self, objects_ids, dual_vertex_center_dev_loss=None):
        regs_losses = []
        for object_id in objects_ids:
            sdf_reg_weight = 0.1  # 0. -> 0.2 in official repo
            # sdf_reg_weight_init  = 0.2
            # sdf_reg_weight = sdf_reg_weight_init - (sdf_reg_weight_init - sdf_reg_weight_init / 20) * min(1.0, 4.0 * t_iter)

            dual_vertex_center_dev_weight = 0.5

            weights_weight = 0.1

            sdf_f1x6x2 = self.fc_sdfs[object_id][
                self.fc_grid_edges.reshape(-1)
            ].reshape(-1, 2)
            mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
            sdf_f1x6x2 = sdf_f1x6x2[mask]
            sdf_diff_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                sdf_f1x6x2[..., 0],
                (sdf_f1x6x2[..., 1] > 0).float(),
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                sdf_f1x6x2[..., 1],
                (sdf_f1x6x2[..., 0] > 0).float(),
            )

            reg_loss = sdf_diff_loss.mean() * sdf_reg_weight

            if (
                self.dual_vertex_center_dev_loss is not None
                and len(self.dual_vertex_center_dev_loss) > 0
            ):
                reg_loss += (
                    self.dual_vertex_center_dev_loss[object_id].mean()
                    * dual_vertex_center_dev_weight
                )  # L_dev as in Equation 8 of our paper
            reg_loss += (
                self.fc_weights[object_id][:, :20]
            ).abs().mean() * weights_weight

            regs_losses.append(reg_loss)

            # sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - FLAGS.sdf_regularizer / 20) * min(1.0,
            #                                                                                                 4.0 * t_iter)
            # reg_loss = loss.sdf_reg_loss(sdf,
            #                              grid_edges).mean() * sdf_weight  # Loss to eliminate internal floaters that are not visible
            # reg_loss += L_dev.mean() * 0.5
            # reg_loss += (weight[:, :20]).abs().mean() * 0.1
            # total_loss = mask_loss + depth_loss + reg_loss
            #
            # if FLAGS.sdf_loss:  # optionally add SDF loss to eliminate internal structures
            #     with torch.no_grad():
            #         pts = sample_random_points(1000, gt_mesh)
            #         gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            #     pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
            #     total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
            #
            # # optionally add developability regularizer, as described in paper section 5.2
            # if FLAGS.develop_reg:
            #     reg_weight = max(0, t_iter - 0.8) * 5
            #     if reg_weight > 0:  # only applied after shape converges
            #         reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
            #         reg_loss += (deform).abs().mean()
            #         reg_loss += (weight[:, :20]).abs().mean()
            #         total_loss = mask_loss + depth_loss + reg_loss
            #
            # regs_losses.append(((self.get_sdf_gradient(object_id=object_id).norm(dim=-1) - 1) ** 2).mean())

        regs_losses = torch.stack(regs_losses)
        return regs_losses

    #
    # @property
    # def verts(self):
    #     return self.verts_params
    #
    # @verts.setter
    # def verts(self, value):
    #     self.verts_params = value
    #
    # def init_sdf(self):
    #     """
    #     """
    #     # calculate distance for each vertex of the tetrahedra to the meshes
    #     # 1. for each vertex, calculate the distance to the faces,
    #     # 2. for each vertex, calculate the distance to the vertices,
    #     # 3. take the minimum distance
    #     # 4. then check if inside for each vertex and determine the sign of the distance
    #
    #
    #     pass
    #
    # @classmethod
    # def read_from_ply_files(
    #     cls,
    #     fpaths_meshes: List[Path],
    #     fpaths_meshes_tforms: List[Path] = None,
    #     device=None,
    #     normalize_scale=False,
    #     **kwargs,
    # ):
    #     meshes = []
    #     for i, fpath_mesh in enumerate(fpaths_meshes):
    #         mesh = Mesh.load_from_file(fpath=fpath_mesh, device=device)
    #         if fpaths_meshes_tforms is not None and fpaths_meshes_tforms[i] is not None:
    #             mesh_tform = torch.load(fpaths_meshes_tforms[i]).to(device)
    #             mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=mesh_tform)
    #         if normalize_scale:
    #             mesh.verts /= mesh.verts.abs().max()
    #         meshes.append(mesh)
    #
    # def get_init_sdf(self, pts):
    #     """
    #         Args:
    #             pts (torch.Tensor): BxMxNx3
    #         Returns:
    #             sdf (torch.Tensor): BxMxN
    #     """
    #     init_radius = 1.
    #     sdf = init_radius - pts.norm(dim=-1, keepdim=True)  # init sdf is a sphere centered at origin
    #     return sdf
    #
    # def get_sdf(self, pts, objects_ids=None):
    #     """
    #     Args:
    #         pts (torch.Tensor): BxNx3
    #     Returns:
    #         sdf (torch.Tensor): BxN
    #     """
    #
    #
    #     B = pts.shape[0]
    #     if objects_ids is None and len(self) > 1:
    #         raise ValueError("objects_ids must be provided when there are multiple objects")
    #
    #     if objects_ids is None and len(self) == 1:
    #         objects_ids
    #
    #     sdf_vals = torch.zeros(pts.shape[:-1], device=pts.device)
    #     for m in range(self.meshes_count):
    #         sdf_vals[m] = self.sdf_mlps[m](pts[m])
    #
    #     return sdf_vals
    #
    # def update_mesh_with_sdf(self, objects_ids=None, jitter_grid=True):
    #     # Run DM tet to get a base mesh
    #     v_deformed = self.verts
    #
    #     if jitter_grid and self.mesh_update_jitter_grid > 0:
    #         jitter = (torch.rand(1, device=v_deformed.device) * 2 - 1) * self.mesh_update_jitter_grid * self.grid_scale
    #         v_deformed = v_deformed + jitter
    #
    #     self.current_sdf = self.get_sdf(v_deformed, total_iter=total_iter)
    #     verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.current_sdf, self.indices)
    #     self.mesh_verts = verts
    #     return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material=None)
    #


"""
    def __init__(self, grid_res, scale, num_layers=None, hidden_size=None, embedder_freq=None, embed_concat_pts=True,
                 init_sdf=None, jitter_grid=0., symmetrize=False):
        super(DMTets, self).__init__()

        self.grid_res = grid_res
        self.marching_tets = DMTet()
        self.grid_scale = scale
        self.init_sdf = init_sdf
        self.jitter_grid = jitter_grid
        self.symmetrize = symmetrize
        self.load_tets(self.grid_res, self.grid_scale)

        embedder_scalar = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
        self.mlp = CoordMLP(
            3,
            1,
            num_layers,
            nf=hidden_size,
            dropout=0,
            activation=None,
            min_max=None,
            n_harmonic_functions=embedder_freq,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=embed_concat_pts)

    def load_tets(self, grid_res=None, scale=None):
        if grid_res is None:
            grid_res = self.grid_res
        else:
            self.grid_res = grid_res
        if scale is None:
            scale = self.grid_scale
        else:
            self.grid_scale = scale
        tets = np.load('data/tets/{}_tets.npz'.format(grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32,
                                  device='cuda') * scale  # verts original scale (-0.5, 0.5)
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

    def get_sdf(self, pts=None, total_iter=0):
        if pts is None:
            pts = self.verts
        if self.symmetrize:
            xs, ys, zs = pts.unbind(-1)
            pts = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        sdf = self.mlp(pts)

        if self.init_sdf is None:
            pass
        elif type(self.init_sdf) in [float, int]:
            sdf = sdf + self.init_sdf
        elif self.init_sdf == 'sphere':
            init_radius = self.grid_scale * 0.25
            init_sdf = init_radius - pts.norm(dim=-1, keepdim=True)  # init sdf is a sphere centered at origin
            sdf = sdf + init_sdf
        elif self.init_sdf == 'ellipsoid':
            rxy = self.grid_scale * 0.15
            xs, ys, zs = pts.unbind(-1)
            init_sdf = rxy - torch.stack([xs, ys, zs / 2], -1).norm(dim=-1,
                                                                    keepdim=True)  # init sdf is approximately an ellipsoid centered at origin
            sdf = sdf + init_sdf
        else:
            raise NotImplementedError

        return sdf

    def get_sdf_gradient(self):
        num_samples = 5000
        sample_points = (torch.rand(num_samples, 3, device=self.verts.device) - 0.5) * self.grid_scale
        mesh_verts = self.mesh_verts.detach() + (torch.rand_like(self.mesh_verts) - 0.5) * 0.1 * self.grid_scale
        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        y = self.get_sdf(pts=sample_points)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients

    def get_sdf_reg_loss(self):
        reg_loss = ((self.get_sdf_gradient().norm(dim=-1) - 1) ** 2).mean()
        return reg_loss

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material=None, total_iter=0, jitter_grid=True):
        # Run DM tet to get a base mesh
        v_deformed = self.verts

        # if self.FLAGS.deform_grid:
        #     v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        # else:
        #     v_deformed = self.verts
        if jitter_grid and self.jitter_grid > 0:
            jitter = (torch.rand(1, device=v_deformed.device) * 2 - 1) * self.jitter_grid * self.grid_scale
            v_deformed = v_deformed + jitter

        self.current_sdf = self.get_sdf(v_deformed, total_iter=total_iter)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.current_sdf, self.indices)
        self.mesh_verts = verts
        return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material)

"""
