import logging
import pickle

logger = logging.getLogger(__name__)
import abc

from od3d_basic.cv.visual.sample import sample_pxl2d_pts

from enum import Enum
from typing import List, Union, Optional

from omegaconf import DictConfig
import inspect
import torch
import torch.nn as nn
from od3d_basic.data.batch_datatypes import OD3D_ModelData


class PROJECT_MODALITIES(str, Enum):
    DEPTH = "depth"
    MASK = "mask"
    MASK_VERTS_VSBL = "mask_verts_vsbl"
    RGB = "rgb"
    RGBA = "rgba"
    FEATS = "feats"
    FEATS_PBR = "feats_pbr"
    FEATS_COARSE = "feats_coarse"
    IMG = "img"
    CLUTTER_PXL2D = "clutter_pxl2d"
    ID = "id"
    ONEHOT = "onehot"
    ONEHOT_SMOOTH = "onehot_smooth"
    ONEHOT_COARSE = "onehot_coarse"
    PT3D = "pt3d"
    NOCS = "nocs"
    CANOCS = "canocs"
    PT3D_COARSE = "pt3d_coarse"
    PT3D_NCDS = "pt3d_ncds"
    PT3D_NCDS_AVG = "pt3d_ncds_avg"
    PXL2D = "pxl2d"
    OBJ_ID = "obj_id"
    OBJ_ONEHOT = "obj_onehot"
    OBJ_IN_SCENE_ID = "obj_in_scene_id"
    OBJ_IN_SCENE_ONEHOT = "obj_in_scene_onehot"
    WIREFRAME = 'wireframe'


class FEATS_DISTR(str, Enum):
    VON_MISES_FISHER = "von-mises-fisher"
    GAUSSIAN = "gaussian"


class FEATS_ACTIVATION(str, Enum):
    NONE = "none"
    NORM = "norm"
    NORM_DETACH = "norm_detach"
    SIGMOID = "sigmoid"
    TANH = "tanh"


# method = OD3D_Objects3D.subclasses[self.config.method.class_name]()
from dataclasses import dataclass
from collections.abc import Sequence

class OD3D_Objects3D_Deform(Sequence):
    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def get_out_dim(self):
        raise NotImplementedError

    @classmethod
    def get_model(cls, instance_deform_net_config: DictConfig):
        from od3d_basic.models.model import OD3D_Model

        return OD3D_Model(instance_deform_net_config)

    @classmethod
    def from_net_output(cls, net_output: OD3D_ModelData, affine=False):
        """
        Args:
            net_output (OD3D_ModelData): feat: BxF, feats: BxNxF, featmap: BxFxHxW
        Returns:
            objects3d_deform (OD3D_Objects3D_Deform): B
        """
        raise NotImplementedError


class OD3D_Objects3D(abc.ABC, nn.Module):
    instance_deform_class = OD3D_Objects3D_Deform
    subclasses = {}
    feat_clutter: Optional[torch.Tensor]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create_from_config(cls, config: DictConfig):
        keys = inspect.getfullargspec(cls.__init__)[0][1:]
        od3d_objects3d = cls(
            **{
                key: config.get(key)
                for key in keys
                if config.get(key, None) is not None
            },
        )
        return od3d_objects3d

    def to_bytes(self):
        b = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        return torch.frombuffer(b, dtype=torch.uint8)

    @classmethod
    def from_bytes(cls, byte_tensor):
        # Ensure it's on CPU and contiguous
        byte_tensor = byte_tensor.cpu().contiguous()

        # Convert back to Python bytes
        b = byte_tensor.numpy().tobytes()

        # Reconstruct original mesh object
        mesh = pickle.loads(b)
        return mesh

    def read_from_files(self):
        raise NotImplementedError

    def read_from_file(self):
        raise NotImplementedError

    def write_to_files(self):
        raise NotImplementedError

    def get_instance_deform(
        self,
        imgs_feats,
        objects_ids=None,
        img_feats_canonical=None,
        use_coord_mlp=False,
        return_latent_feature=False,
    ):
        """
        Args:
            imgs_feats (torch.Tensor): BxCxHxW, BxC
        """

        if self.instance_deform_net is not None:
            if (
                self.instance_deform_net.config.get("in_feats", "backbone")
                == "backbone"
            ):
                pass
            elif self.instance_deform_net.config.get("in_feats", "backbone") == "head":
                imgs_feats.featmaps[-1] = img_feats_canonical
            elif (
                self.instance_deform_net.config.get("in_feats", "backbone")
                == "head_detach"
            ):
                imgs_feats.featmaps[-1] = img_feats_canonical.detach()

            from od3d_basic.models.heads.coordmlp import CoordMLP

            if isinstance(self.instance_deform_net.head, CoordMLP):
                imgs_feats.pts3d = self.sample(
                    cams_tform4x4_obj=None,
                    cams_intr4x4=None,
                    imgs_sizes=None,
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.PT3D,
                    broadcast_batch_and_cams=False,
                    down_sample_rate=1.0,
                    add_clutter=False,
                    add_other_objects=False,
                    dtype=None,
                    device=None,
                    sample_clutter=False,
                    sample_other_objects=False,
                    instance_deform=None,
                ).detach()  # B x 3 x N

            nearest_pt3d = self.sample_nearest_to_feats2d_img(
                feats2d_img=img_feats_canonical,
                objects_ids=objects_ids,
                modalities=PROJECT_MODALITIES.PT3D,
                add_clutter=True,
                instance_deform=None,
            )
            from od3d_basic.cv.visual.resize import resize

            nearest_pt3d = resize(
                nearest_pt3d,
                scale_factor=imgs_feats.featmaps[-1].shape[-1] / nearest_pt3d.shape[-1],
                mode="nearest_v2",
            )

            instance_deform_net_nearest_pt3d = self.instance_deform_net.config.get(
                "nearest_pt3d",
                "cat",
            )
            if instance_deform_net_nearest_pt3d == "cat":
                imgs_feats.featmaps[-1] = torch.cat(
                    (imgs_feats.featmaps[-1], nearest_pt3d),
                    dim=1,
                )
            elif instance_deform_net_nearest_pt3d == "cat_harmonics":
                from od3d_basic.models.heads.coordmlp.head import HarmonicEmbedding

                harmonic_embedding = HarmonicEmbedding(dim=1)
                imgs_feats.featmaps[-1] = torch.cat(
                    (imgs_feats.featmaps[-1], harmonic_embedding(nearest_pt3d)),
                    dim=1,
                )
            elif instance_deform_net_nearest_pt3d == "sole":
                imgs_feats.featmaps[-1] = nearest_pt3d
            elif instance_deform_net_nearest_pt3d == "sole_harmonics":
                from od3d_basic.models.heads.coordmlp.head import HarmonicEmbedding

                harmonic_embedding = HarmonicEmbedding(dim=1)
                imgs_feats.featmaps[-1] = harmonic_embedding(nearest_pt3d)
            else:
                pass

            affine = self.instance_deform_net.config.get("affine", False)
            net_out = self.instance_deform_net(imgs_feats)
            if affine:
                meshes_deform = self.instance_deform_class.from_net_output(
                    net_out,
                    affine=affine,
                )
                meshes_deform.verts_deform[..., :3] = (
                    meshes_deform.verts_deform[..., :3] + 1.0
                )

                verts = self.sample(
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.PT3D,
                    add_clutter=False,
                    instance_deform=None,
                ).detach()

                meshes_deform.verts_deform = (
                    meshes_deform.verts_deform[..., :3] * verts
                    + meshes_deform.verts_deform[..., 3:]
                    - verts
                )
            else:
                meshes_deform = self.instance_deform_class.from_net_output(
                    net_out,
                    affine=affine,
                )

            return meshes_deform
        else:
            return None

    def __init__(
        self,
        feat_dim=128,
        objects_count=0,
        feat_clutter=False,
        feats_objects=False,
        feats_requires_grad=True,
        feat_clutter_requires_param=True,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        feats_activation=FEATS_ACTIVATION.NORM_DETACH,
        verts_requires_grad=False,
        device=None,
        dtype=None,
        scale_3D_params=1e2,
        instance_deform_net_config: DictConfig = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.feat_dim = feat_dim
        self.objects_count = objects_count
        self.scale_3D_params = scale_3D_params
        self.feats_activation = feats_activation

        self.verts_requires_grad = verts_requires_grad
        self.feat_clutter_requires_param = feat_clutter_requires_param

        if feat_clutter_requires_param:
            if isinstance(feat_clutter, torch.Tensor) or feat_clutter:
                self._feat_clutter = torch.nn.Parameter(
                    torch.empty(self.feat_dim, **factory_kwargs),
                    requires_grad=feats_requires_grad,
                )  # F,
            else:
                self.register_parameter("_feat_clutter", None)  # None
        else:
            self._feat_clutter = torch.zeros(self.feat_dim, **factory_kwargs)  # F,

        if instance_deform_net_config is not None:
            self.instance_deform_net = self.instance_deform_class.get_model(
                instance_deform_net_config,
            )
            self.verts_deform_requires_grad = (
                True  # self.instance_deform_net.requires_grad_()
            )
        else:
            self.instance_deform_net = None
            self.verts_deform_requires_grad = False

        self.feats_distribution = feats_distribution

        if instance_deform_net_config is not None:
            affine = instance_deform_net_config.get("affine", False)
            if instance_deform_net_config.head.get("class_name", "MLP") == "CoordMLP":
                if not affine:
                    instance_deform_net_config.head.update({"out_dim": 3})
                else:
                    instance_deform_net_config.head.update({"out_dim": 6})
            else:
                if not affine:
                    instance_deform_net_config.head.update(
                        {"out_dim": 3 * self.verts_counts_max},
                    )
                else:
                    instance_deform_net_config.head.update(
                        {"out_dim": 6 * self.verts_counts_max},
                    )

    def reset_parameters(self, feat_clutter: Union[bool, torch.Tensor] = False) -> None:
        if self.feat_clutter_requires_param:
            if self._feat_clutter is not None:
                if isinstance(feat_clutter, torch.Tensor):
                    with torch.no_grad():
                        self._feat_clutter.copy_(feat_clutter)
                else:
                    torch.nn.init.normal_(self._feat_clutter)

    def set_verts_requires_grad(self, verts_requires_grad):
        self.verts_requires_grad = verts_requires_grad

    def set_verts_deform_requires_grad(self, verts_deform_requires_grad):
        self.verts_deform_requires_grad = verts_deform_requires_grad

    @property
    def feat_clutter(self):
        return self.activate_feats(self._feat_clutter)

    @feat_clutter.setter
    def feat_clutter(self, value):
        if self._feat_clutter is None or not isinstance(
            self._feat_clutter,
            torch.nn.Parameter,
        ):
            self._feat_clutter = value
        else:
            with torch.no_grad():
                _ = self._feat_clutter.copy_(value)

    def activate_feats(self, feats):
        if feats is not None:
            if self.feats_activation == FEATS_ACTIVATION.NORM_DETACH:
                feats = feats / ((feats.norm(dim=-1, keepdim=True) + 1e-10).detach())
            elif self.feats_activation == FEATS_ACTIVATION.NORM:
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)
            elif self.feats_activation == FEATS_ACTIVATION.SIGMOID:
                feats = torch.nn.functional.sigmoid(feats)
            elif self.feats_activation == FEATS_ACTIVATION.NONE:
                pass
            elif self.feats_activation == FEATS_ACTIVATION.TANH:
                feats = torch.nn.functional.tanh(feats)
        return feats

    def __len__(self):
        return self.objects_count

    def cams_downsample(self, cams_intr4x4=None, imgs_sizes=None, down_sample_rate=1.0):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
        Returns:
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
        """
        if down_sample_rate != 1.0:
            if cams_intr4x4 is not None:
                cams_intr4x4 = cams_intr4x4.clone()
                if cams_intr4x4.dim() == 2:
                    cams_intr4x4[:2] /= down_sample_rate
                elif cams_intr4x4.dim() == 3:
                    cams_intr4x4[:, :2] /= down_sample_rate
                elif cams_intr4x4.dim() == 4:
                    cams_intr4x4[:, :, :2] /= down_sample_rate
                else:
                    raise NotImplementedError

            if imgs_sizes is not None:
                if isinstance(imgs_sizes, torch.Size):
                    imgs_sizes = torch.LongTensor(list(imgs_sizes))
                imgs_sizes = imgs_sizes.clone() // down_sample_rate
        else:
            if cams_intr4x4 is not None:
                cams_intr4x4 = cams_intr4x4.clone()
            if imgs_sizes is not None:
                if isinstance(imgs_sizes, torch.Size):
                    imgs_sizes = torch.LongTensor(list(imgs_sizes))
                imgs_sizes = imgs_sizes.clone()
        return cams_intr4x4, imgs_sizes

    def cams_and_objects_broadcast(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids,
        instance_deform=None,
        obj_tform4x4_objs=None,
    ):
        scenes_count = objects_ids.shape[0]  # sc
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f"Set `cams_tform4x4_obj.dim()` must be 3 or 4")

        objects_ids = objects_ids.clone()
        if cams_tform4x4_obj.dim() == 3:
            cams_tform4x4_obj = cams_tform4x4_obj[None, :]
        if cams_intr4x4.dim() == 3:
            cams_intr4x4 = cams_intr4x4[None, :]
        cams_tform4x4_obj = cams_tform4x4_obj.expand(
            scenes_count,
            cams_count,
            4,
            4,
        ).reshape(-1, 4, 4)
        cams_intr4x4 = cams_intr4x4.expand(scenes_count, cams_count, 4, 4).reshape(
            -1,
            4,
            4,
        )

        if objects_ids.dim() == 1:
            objects_per_scene_count = 1
            objects_ids = (
                objects_ids[:, None].expand(scenes_count, cams_count).reshape(-1)
            )
        elif objects_ids.dim() == 2:
            objects_per_scene_count = objects_ids.shape[-1]
            objects_ids = (
                objects_ids[:, None]
                .expand(scenes_count, cams_count, objects_per_scene_count)
                .reshape(-1, objects_per_scene_count)
            )
        else:
            msg = f"no implementation for dim of objects ids: {objects_ids.dim()}"
            raise NotImplementedError(msg)

        if obj_tform4x4_objs is not None:
            if obj_tform4x4_objs.dim() == 3:
                obj_tform4x4_objs = (
                    obj_tform4x4_objs[:, None, None]
                    .expand(
                        scenes_count,
                        cams_count,
                        objects_per_scene_count,
                        4,
                        4,
                    )
                    .reshape(-1, objects_per_scene_count, 4, 4)
                )
            elif obj_tform4x4_objs.dim() == 4:
                objects_per_scene_count = obj_tform4x4_objs.shape[1]
                obj_tform4x4_objs = (
                    obj_tform4x4_objs[:, None]
                    .expand(
                        scenes_count,
                        cams_count,
                        objects_per_scene_count,
                        4,
                        4,
                    )
                    .reshape(-1, objects_per_scene_count, 4, 4)
                )
            else:
                msg = f"no implementation for dim of obj_tform4x4_objs: {obj_tform4x4_objs.dim()}"
                raise NotImplementedError(msg)

        if instance_deform is not None:
            instance_deform = instance_deform.broadcast_with_cams(cams_count)

        return (
            cams_tform4x4_obj,
            cams_intr4x4,
            objects_ids,
            instance_deform,
            obj_tform4x4_objs,
        )

    def forward(self):
        pass

    def update_verts(self, require_grad=True):
        pass

    def update_verts_deform(self, batch, imgs_feats, require_grad=True):
        if not require_grad:
            with torch.no_grad():
                self.instance_deform = self.get_instance_deform(
                    imgs_feats=imgs_feats,
                    objects_ids=batch.categories_ids,
                )
        else:
            self.instance_deform = self.get_instance_deform(
                imgs_feats=imgs_feats,
                objects_ids=batch.categories_ids,
            )

    def update_and_render_with_batch_and_imgs_feats(
        self,
        batch,
        imgs_feats=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        down_sample_rate=1.0,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
        verts_requires_grad=True,
        verts_deform_requires_grad=True,
        broadcast_batch_and_cams=False,
        rgb_light_env=None,
        add_clutter=True,
    ):
        if self.verts_requires_grad:
            self.update_verts(require_grad=verts_requires_grad)
        if self.verts_deform_requires_grad:
            self.update_verts_deform_with_batch(
                batch=batch,
                imgs_feats=imgs_feats,
                require_grad=verts_deform_requires_grad,
            )

        return self.render_with_batch(
            batch=batch,
            modalities=modalities,
            down_sample_rate=down_sample_rate,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            rgb_light_env=rgb_light_env,
            add_clutter=add_clutter,
        )

    def render_with_batch(
        self,
        batch,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        down_sample_rate=1.0,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
        broadcast_batch_and_cams=False,
        rgb_light_env=None,
        add_clutter=True,
    ):
        if batch.objects_ids is not None:
            objects_ids = batch.objects_ids
        else:
            objects_ids = torch.LongTensor(list(range(len(self)))).to(
                device=batch.cam_tform4x4_obj.device,
            )
            objects_ids = objects_ids[:, None]

        return self.render(
            modalities=modalities,
            cams_tform4x4_obj=batch.cam_tform4x4_obj,
            cams_intr4x4=batch.cam_intr4x4,
            imgs_sizes=batch.size,
            objects_ids=objects_ids,
            down_sample_rate=down_sample_rate,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            add_clutter=add_clutter,
            obj_tform4x4_objs=batch.obj_tform4x4_objs,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            rgb_light_env=rgb_light_env,
        )

    def render(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        instance_deform=None,
        max_batch_size=None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
        **kwargs,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W) , or tuple
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the rendered features.
            add_other_objects: bool, determines wether to add other objects' features to the rendered features.
            instance_deform (OD3D_Objects3D_Deform): the instance deformation to apply (B x ...).
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        self()

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        # imgs_size: (height, width)
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        self.to(device)

        if not isinstance(imgs_sizes, torch.Tensor):
            imgs_sizes = torch.LongTensor(imgs_sizes).to(device)

        cams_intr4x4, imgs_sizes = self.cams_downsample(
            cams_intr4x4,
            imgs_sizes,
            down_sample_rate,
        )
        if obj_tform4x4_objs is not None and isinstance(obj_tform4x4_objs, list):
            obj_tform4x4_objs = torch.stack(obj_tform4x4_objs)

        if objects_ids is None:
            objects_ids = torch.LongTensor(list(range(len(self)))).to(device=device)
            if obj_tform4x4_objs is not None and obj_tform4x4_objs.dim() > 3:
                objects_ids = objects_ids.reshape(
                    *(((1,) * (obj_tform4x4_objs.dim() - 3)) + objects_ids.shape),
                )
                objects_ids = objects_ids.expand(*obj_tform4x4_objs.shape[:-2])
                # *(objs_lengths_acc_from_0.shape + ((1,) * (objs_mod.dim() - objs_lengths.dim())
        elif isinstance(objects_ids, int):
            objects_ids = torch.LongTensor([objects_ids]).to(device=device)
        elif isinstance(objects_ids, List):
            objects_ids = torch.LongTensor(objects_ids).to(device=device)
        elif isinstance(objects_ids, torch.LongTensor):
            objects_ids = objects_ids.clone().to(device=device)

        if broadcast_batch_and_cams:
            objects_count = objects_ids.shape[0]

            (
                cams_tform4x4_obj,
                cams_intr4x4,
                objects_ids,
                instance_deform,
                obj_tform4x4_objs,
            ) = self.cams_and_objects_broadcast(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                objects_ids=objects_ids,
                instance_deform=instance_deform,
                obj_tform4x4_objs=obj_tform4x4_objs,
            )

            cams_times_objects_count = cams_tform4x4_obj.shape[0]
            cams_count = int(cams_times_objects_count / objects_count)
        else:
            objects_count = objects_ids.shape[0]
            if cams_tform4x4_obj is not None:
                cams_count = cams_tform4x4_obj.shape[0]
            else:
                cams_count = objects_count

            if objects_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and objects",
                )

        mods2d_rendered = self.render_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=_modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            obj_tform4x4_objs=obj_tform4x4_objs,
            **kwargs,
        )

        if broadcast_batch_and_cams:
            for key, val in mods2d_rendered.items():
                mods2d_rendered[key] = val.reshape(
                    objects_count,
                    cams_count,
                    *val.shape[1:],
                )

        if not isinstance(modalities, List):
            return mods2d_rendered[_modalities[0]]
        else:
            return mods2d_rendered

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
        instance_deform: OD3D_Objects3D_Deform = None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
        **kwargs,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            add_clutter: bool, determines whether to add clutter features to the rendered features.
            add_other_objects: bool, determines whether to add other objects' features to the rendered features.
            instance_deform: (OD3D_Objects3D_Deform) the instance deformation to apply (B x ...).
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        raise NotImplementedError

    def sample_with_img2d(
        self,
        img2d=None,
        img2d_mask=None,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        sample_clutter_count=0,
        clutter_pxl2d=None,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
    ):
        """
        Sample the objects' projection.
        Args:
            img2d: (B, C, H, W) tensor of the image.
            img2d_mask: (B, 1, H, W) tensor of the image mask.
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.
            instance_deform: (OD3D_Objects3D_Deform) the instance deformation to apply (B x ...).
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, C, V, F), or (B, V, F) dict of projected modalities.
        """

        if dtype is None and cams_tform4x4_obj is not None:
            dtype = cams_tform4x4_obj.dtype
        if device is None and cams_tform4x4_obj is not None:
            device = cams_tform4x4_obj.device

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        sample_modalities = _modalities.copy()
        if PROJECT_MODALITIES.IMG in sample_modalities:
            sample_modalities.remove(PROJECT_MODALITIES.IMG)
        if PROJECT_MODALITIES.CLUTTER_PXL2D in sample_modalities:
            sample_modalities.remove(PROJECT_MODALITIES.CLUTTER_PXL2D)
        if PROJECT_MODALITIES.MASK not in sample_modalities:
            sample_modalities.append(PROJECT_MODALITIES.MASK)
        if PROJECT_MODALITIES.PXL2D not in sample_modalities:
            sample_modalities.append(PROJECT_MODALITIES.PXL2D)

        mods1d_sampled = self.sample(
            modalities=sample_modalities,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            dtype=dtype,
            device=device,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
        )

        feats1d_obj_mask, feats1d_obj_pxl2d = (
            mods1d_sampled[PROJECT_MODALITIES.MASK],
            mods1d_sampled[PROJECT_MODALITIES.PXL2D],
        )

        if sample_clutter_count > 0:
            feats2d_obj_mask = self.render(
                modalities=PROJECT_MODALITIES.MASK,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=False,
                add_other_objects=False,
                instance_deform=instance_deform,
                detach_objects=detach_objects,
                detach_deform=detach_deform,
            )
            if img2d_mask is not None:
                feats2d_clutter_mask = 1.0 - (feats2d_obj_mask * img2d_mask)
            else:
                feats2d_clutter_mask = 1.0 - feats2d_obj_mask

            H, W = img2d_mask.shape[-2:]
            xy = torch.stack(
                torch.meshgrid(
                    torch.arange(W, device=device),
                    torch.arange(H, device=device),
                    indexing="xy",
                ),
                dim=0,
            )  # HxW
            prob_noise = feats2d_clutter_mask.clamp(0, 1).flatten(1)
            prob_noise[prob_noise.sum(dim=-1) <= 0.0] = 1.0
            clutter_pxl2d = xy.flatten(1)[
                :,
                torch.multinomial(prob_noise, sample_clutter_count, replacement=True),
            ].permute(1, 2, 0)

        if clutter_pxl2d is not None:
            feats1d_obj_pxl2d = torch.cat([feats1d_obj_pxl2d, clutter_pxl2d], dim=1)
            B = feats1d_obj_mask.shape[0]
            sample_clutter_count = clutter_pxl2d.shape[1]
            feats1d_clutter_mask = torch.ones(
                (B, sample_clutter_count),
                dtype=torch.bool,
                device=feats1d_obj_mask.device,
            )
            feats1d_obj_mask = torch.cat(
                [feats1d_obj_mask, feats1d_clutter_mask],
                dim=-1,
            )

        # mods1d_sampled = {}
        for modality in _modalities:
            # feats, mask, labels
            if modality == PROJECT_MODALITIES.IMG:
                mods1d_sampled[modality] = sample_pxl2d_pts(img2d, feats1d_obj_pxl2d)
            elif modality == PROJECT_MODALITIES.PXL2D:
                mods1d_sampled[modality] = feats1d_obj_pxl2d
            elif modality == PROJECT_MODALITIES.CLUTTER_PXL2D:
                mods1d_sampled[modality] = clutter_pxl2d
            elif modality == PROJECT_MODALITIES.MASK:
                mods1d_sampled[modality] = feats1d_obj_mask
            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
                or modality == PROJECT_MODALITIES.ONEHOT_COARSE
            ):
                B = mods1d_sampled[modality].shape[0]
                V = mods1d_sampled[modality].shape[1]
                label1d_clutter = self.get_label_clutter(
                    add_other_objects=add_other_objects,
                    one_hot=True,
                    device=device,
                    coarse=modality == PROJECT_MODALITIES.ONEHOT_COARSE,
                )[:, :, None].expand(
                    B,
                    V,
                    sample_clutter_count,
                )
                mods1d_sampled[modality] = torch.cat(
                    [mods1d_sampled[modality], label1d_clutter],
                    dim=-1,
                )
            elif modality == PROJECT_MODALITIES.ID:
                B = mods1d_sampled[modality].shape[0]
                label1d_clutter = self.get_label_clutter(
                    add_other_objects=add_other_objects,
                    one_hot=False,
                    device=device,
                )[
                    None,
                ].expand(
                    B,
                    sample_clutter_count,
                )
                mods1d_sampled[modality] = torch.cat(
                    [mods1d_sampled[modality], label1d_clutter],
                    dim=-1,
                )
            elif modality == PROJECT_MODALITIES.FEATS:
                mods1d_sampled[modality] = mods1d_sampled[modality]
                # mods1d_sampled[modality] = torch.cat([mods1d_sampled[modality], self.feat_clutter[None, :].expand(
                #    mods1d_sampled[modality].shape[0], sample_clutter_count, self.feat_dim)], dim=-2)

            elif modality == PROJECT_MODALITIES.FEATS_COARSE:
                mods1d_sampled[modality] = mods1d_sampled[modality]
                # mods1d_sampled[modality] = torch.cat([mods1d_sampled[modality], self.feat_clutter[None, :].expand(
                #    mods1d_sampled[modality].shape[0], sample_clutter_count, self.feat_dim)], dim=-2)

            else:
                raise ValueError(f"Unknown modality {modality}")

        if not isinstance(modalities, List):
            return mods1d_sampled[_modalities[0]]
        else:
            return mods1d_sampled

    def sample(
        self,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        sample_clutter=False,
        sample_other_objects=False,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[PROJECT_MODALITIES, List(PROJECT_MODALITIES)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.
            obj_tform4x4_objs: (O, 4, 4) or (B, O, 4, 4)
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, C, V, F), or (B, V, F) dict of projected modalities.
        """

        self()

        if not isinstance(modalities, List):
            _modalities = [modalities]
        else:
            _modalities = modalities

        # imgs_size: (height, width)
        if dtype is None and cams_tform4x4_obj is not None:
            dtype = cams_tform4x4_obj.dtype
        if device is None and cams_tform4x4_obj is not None:
            device = cams_tform4x4_obj.device

        self.to(device)

        cams_intr4x4, imgs_sizes = self.cams_downsample(
            cams_intr4x4,
            imgs_sizes,
            down_sample_rate,
        )

        if objects_ids is None:
            objects_ids = torch.LongTensor(list(range(len(self)))).to(device=device)
        elif isinstance(objects_ids, int):
            objects_ids = torch.LongTensor([objects_ids]).to(device=device)
        elif isinstance(objects_ids, List):
            objects_ids = torch.LongTensor(objects_ids).to(device=device)
        elif isinstance(objects_ids, torch.LongTensor):
            objects_ids = objects_ids.clone().to(device=device)

        if broadcast_batch_and_cams:
            objects_count = objects_ids.shape[0]

            (
                cams_tform4x4_obj,
                cams_intr4x4,
                objects_ids,
                instance_deform,
                obj_tform4x4_objs,
            ) = self.cams_and_objects_broadcast(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                objects_ids=objects_ids,
                instance_deform=instance_deform,
                obj_tform4x4_objs=obj_tform4x4_objs,
            )

            cams_times_objects_count = cams_tform4x4_obj.shape[0]
            cams_count = int(cams_times_objects_count / objects_count)
        else:
            objects_count = objects_ids.shape[0]
            if cams_tform4x4_obj is not None:
                cams_count = cams_tform4x4_obj.shape[0]
            else:
                cams_count = objects_count

            if objects_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and objects",
                )

        mods1d_rendered = self.sample_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=_modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            device=device,
            dtype=dtype,
            sample_other_objects=sample_other_objects,
            sample_clutter=sample_clutter,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            obj_tform4x4_objs=obj_tform4x4_objs,
        )

        if broadcast_batch_and_cams:
            for key, val in mods1d_rendered.items():
                mods1d_rendered[key] = val.reshape(
                    objects_count,
                    cams_count,
                    *val.shape[1:],
                )

        if not isinstance(modalities, List):
            return mods1d_rendered[_modalities[0]]
        else:
            return mods1d_rendered

    def sample_batch(
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
        device=None,
        dtype=None,
        sample_clutter=False,
        sample_other_objects=False,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
        obj_tform4x4_objs=None,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.

        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, V, F) dict of projected modalities.
        """
        raise NotImplementedError

    # def get_sim_project_sample(self, ):

    def get_sim_render(
        self,
        feats2d_img,
        cams_tform4x4_obj,
        cams_intr4x4,
        depth=None,
        depth_mask=None,
        world_scale1d_type=None, # 'none', 'median', 'mean',
        world_scale1d=None,
        objects_scales3d=None,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        allow_clutter=True,
        return_sim_pxl=False,
        add_clutter=False,
        add_other_objects=False,
        temp=1.0,
        instance_deform=None,
        normalize_surface=False,
        object_mask=None,
        use_neg_mse=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats2d_img_mask (torch.Tensor): Bx1xHxW
            cams_tform4x4_obj: (B, C, 4, 4) or (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (4, 4), or (B, 4, 4) or (B, C, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (Union[str, List(str)]) the modalities to render.
            broadcast_batch_and_cams: bool, whether to broadcast the batch and camera dimensions.
            down_sample_rate: float, the downsample rate for the rendered image.
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
            add_other_objects: bool, determines wether to add other objects' features to the sampled features.
            instance_deform (OD3D_Objects3D_Deform): the instance deformation to apply (B x ...).
        Returns:
            sim (torch.Tensor): Bx(C)
            sim_feats2d (torch.Tensor, optional): Bx(C)xHxW
        """
        imgs_sizes = torch.LongTensor(list(feats2d_img.shape[-2:])) * down_sample_rate

        if depth is None:
            if object_mask is not None:
                mods = [PROJECT_MODALITIES.FEATS, PROJECT_MODALITIES.MASK]
            else:
                mods = [PROJECT_MODALITIES.FEATS]
        else:
            mods = [PROJECT_MODALITIES.FEATS, PROJECT_MODALITIES.MASK, PROJECT_MODALITIES.DEPTH]

        mods_rendered = self.render(
            modalities=mods,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            objects_scales3d=objects_scales3d,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            instance_deform=instance_deform,
        )

        if len(mods) == 1:
            feats2d_rendered = mods_rendered[PROJECT_MODALITIES.FEATS]
            mask_rendered = None
        else:
            feats2d_rendered = mods_rendered[PROJECT_MODALITIES.FEATS]
            mask_rendered = mods_rendered[PROJECT_MODALITIES.MASK]

        sim = self.get_sim_feats2d_img_and_rendered(
            feats2d_img,
            feats2d_rendered,
            return_sim_pxl=True,
            feats2d_img_mask=feats2d_img_mask,
            allow_clutter=allow_clutter,
            temp=temp,
            objects_ids=objects_ids,
            normalize_surface=normalize_surface,
            use_neg_mse=use_neg_mse,
        )

        # feats_surface = self.sample(
        #         modalities=PROJECT_MODALITIES.FEATS_COARSE,
        #         objects_ids=objects_ids,
        #         add_clutter=True,
        #     )
        # reduction='mean'
        # net_prob_surface2d_logits = self.get_sim(
        #     "bchw,bfc->bfhw",
        #     feats2d_img,
        #     feats_surface,
        #     temp=temp,
        # )
        #
        # if feats2d_rendered.dim() == 5:
        #     _feats2d_img_mask = feats2d_img_mask[:, None].clone()
        #     net_prob_surface2d_logits = net_prob_surface2d_logits[:, None]
        #     render_prob_surface2d_logits = self.get_sim(
        #         "bkchw,bfc->bkfhw",
        #         feats2d_rendered,
        #         feats_surface,
        #         temp=temp,
        #     )
        # else:
        #     _feats2d_img_mask = feats2d_img_mask.clone()
        #     render_prob_surface2d_logits = self.get_sim(
        #         "bchw,bfc->bfhw",
        #         feats2d_rendered,
        #         feats_surface,
        #         temp=temp,
        #     )
        #
        # # _feats2d_img_mask  *= net_prob_surface2d_logits[..., -1:, :, :] > net_prob_surface2d_logits[..., :-1, :, :]
        # net_prob_surface2d_prob = net_prob_surface2d_logits[..., :, :, :].softmax(dim=-3)
        # net_prob_surface2d_prob_bg = net_prob_surface2d_prob[..., -1:, :, :]
        # render_prob_surface2d_logits_bg = torch.zeros_like(render_prob_surface2d_logits)
        # render_prob_surface2d_logits_bg[..., -1:, :, :] = 1.
        # render_prob_surface2d_logits = (1. - net_prob_surface2d_prob_bg) * render_prob_surface2d_logits + net_prob_surface2d_prob_bg * render_prob_surface2d_logits_bg
        #
        # render_prob_surface2d_prob = render_prob_surface2d_logits[..., :, :, :].softmax(dim=-3)
        # ce_pxl = -(render_prob_surface2d_prob * torch.log(net_prob_surface2d_prob)).mean(dim=-3)
        #sim_pxl = -ce_pxl
        #sim = (sim_pxl * _feats2d_img_mask[..., 0, :, :]).flatten(-2).sum(dim=-1) / ((_feats2d_img_mask[..., 0, :, :]).flatten(-2).sum(dim=-1) + 1e-8)
        # if return_sim_pxl:
        #     sim =  (sim, sim_pxl)
        # else:
        #     sim = sim

        # # sim[1]: BxKxHxW, BxHxW
        # sim_pxl = sim[1]
        # # shape (B, 1, H, W)
        # sim_clutter2d = self.get_sim(
        #     "bchw,c->bhw",
        #     feats2d_img,
        #     self.feat_clutter,
        #     temp=temp,
        #     use_neg_mse=use_neg_mse,
        # )
        # if sim_pxl.dim() == 4:
        #     sim_clutter2d = sim_clutter2d[:, None]
        #
        # if feats2d_img_mask is not None:
        #     if feats2d_rendered.dim() == 5:
        #         _feats2d_img_mask = feats2d_img_mask[:, None].clone()
        #
        #     clutter_mask = (_feats2d_img_mask < 0.5).expand(sim_feats2d.shape)
        # else:
        #     clutter_mask = torch.zeros_like(sim_feats2d, dtype=torch.bool)
        #
        # if allow_clutter:
        #     clutter_mask = clutter_mask | (sim_feats2d < sim_clutter2d)

        # weight = feats2d_img_mask
        if depth is not None:
            depth_rendered = mods_rendered[PROJECT_MODALITIES.DEPTH]

            from od3d_basic.cv.visual.resize import resize_nearest_v2, resize
            depth_res = resize_nearest_v2(
                depth,
                H_out=depth_rendered.shape[-2],
                W_out=depth_rendered.shape[-1],
            )

            depth_mse_to_sim_factor = -0.01

            if object_mask is not None:
                object_mask_res = resize(
                    object_mask,
                    H_out=mask_rendered.shape[-2],
                    W_out=mask_rendered.shape[-1],
                )
            else:
                object_mask_res = None

            #from od3d_basic.cv.visual.show import show_imgs
            #show_imgs(_depths)

            if depth_rendered.dim() == 5:
                depth_res_scaled = depth_res[: , None].clone()
                depth_mask =  (depth_res[:, None] > 0.) * (depth_rendered > 0.)
                if object_mask_res is not None:
                    depth_mask = depth_mask * (object_mask_res[:, None] > 0.5)

                #from od3d_basic.cv.visual.show import show_imgs
                #show_imgs(depth_mask[:, 0] * 1000)
                #show_imgs(sim[1][:, :1])
                if world_scale1d_type == 'median' or world_scale1d_type == 'mean':
                    if world_scale1d is None:
                        world_scale1d = torch.ones_like(depth_rendered[..., 0, 0, 0])

                    objects_to_depth_scale1d_pxl = depth_res[:, None] / (depth_rendered + 1e-8)
                    if world_scale1d_type == 'mean':
                        objects_to_depth_scale1d_mean = ((objects_to_depth_scale1d_pxl * depth_mask).flatten(-3).sum(dim=-1) + 1e-8) / ((depth_mask).flatten(-3).sum(dim=-1) + 1e-8)
                        world_scale1d[:] = objects_to_depth_scale1d_mean.detach().clone()
                    else:
                        objects_to_depth_scale1d_pxl[~depth_mask] = torch.nan
                        objects_to_depth_scale1d_median = objects_to_depth_scale1d_pxl.flatten(-3).nanmedian(dim=-1).values
                        objects_to_depth_scale1d_median = objects_to_depth_scale1d_median.nan_to_num(nan=1., posinf=1., neginf=1.)
                        world_scale1d[:] = objects_to_depth_scale1d_median.detach().clone()
                    depth_res_scaled = depth_res_scaled / world_scale1d[..., None, None, None]
                elif world_scale1d is not None:
                    depth_res_scaled = depth_res_scaled / world_scale1d[..., None, None, None]
                # b_real_scale_norm =
                #depth_mask = depth_mask * torch.softmax((depth_mask * sim[1][:, :, None]).flatten(3), dim=-1).reshape(*depth_mask.shape)
                sim_depth_pxl = depth_mse_to_sim_factor * (torch.sqrt((depth_res_scaled - depth_rendered) ** 2) * depth_mask)
                sim_depth = (sim_depth_pxl.flatten(2).sum(dim=-1) + 1e-8)  / (depth_mask.flatten(2).sum(dim=-1) + 1e-8)
                sim_depth[depth_mask.flatten(2).sum(dim=-1) == 0.] -= 9999. # never select the one where the depth mask Is 0
                sim_depth_pxl = sim_depth_pxl[:, :, 0]
            else:
                depth_res_scaled = depth_res.clone()
                depth_mask = (depth_res > 0.) * (depth_rendered > 0.)
                if object_mask_res is not None:
                    depth_mask = depth_mask *  (object_mask_res > 0.5)

                if world_scale1d_type == 'median' or world_scale1d_type == 'mean':
                    if world_scale1d is None:
                        world_scale1d = torch.ones_like(depth_rendered[..., 0, 0, 0])
                    objects_to_depth_scale1d_pxl = depth_res / (depth_rendered + 1e-8)
                    if world_scale1d_type == 'mean':
                        objects_to_depth_scale1d_mean = ((objects_to_depth_scale1d_pxl * depth_mask).flatten(-3).sum(dim=-1) + 1e-8) / ((depth_mask).flatten(-3).sum(dim=-1) + 1e-8)
                        world_scale1d[:] = objects_to_depth_scale1d_mean.detach().clone()
                    else:
                        objects_to_depth_scale1d_pxl[~depth_mask] = torch.nan
                        objects_to_depth_scale1d_median = objects_to_depth_scale1d_pxl.flatten(-3).nanmedian(dim=-1).values
                        objects_to_depth_scale1d_median = objects_to_depth_scale1d_median.nan_to_num(nan=1., posinf=1., neginf=1.)
                        world_scale1d[:] = objects_to_depth_scale1d_median.detach().clone()
                    depth_res_scaled = depth_res_scaled / world_scale1d[..., None, None, None]
                elif world_scale1d is not None:
                    depth_res_scaled = depth_res_scaled / world_scale1d[..., None, None, None]


                #depth_mask = depth_mask * torch.softmax((depth_mask * sim[1][:, None]).flatten(2), dim=-1).reshape(*depth_mask.shape)

                #from od3d_basic.cv.visual.show import show_imgs
                #show_imgs(depth_mask)
                #_depths = torch.stack([depth_res * depth_mask, depth_rendered * depth_mask], dim=0)
                sim_depth_pxl = depth_mse_to_sim_factor * (torch.sqrt((depth_res_scaled - depth_rendered) ** 2) * depth_mask)
                sim_depth = (sim_depth_pxl.flatten(1).sum(dim=-1) + 1e-8) / (depth_mask.flatten(1).sum(dim=-1) + 1e-8)
                sim_depth_pxl = sim_depth_pxl[:, 0]


            if return_sim_pxl:
                sim = (sim[0] + sim_depth, sim[1] + sim_depth_pxl)
            else:
                sim = sim[0] + sim_depth

        else:
            if return_sim_pxl:
                sim = (sim[0], sim[1])
            else:
                sim = sim[0]

        # if object_mask is not None:
        #     from od3d_basic.cv.visual.resize import resize
        #
        #     object_mask_res = resize(
        #         object_mask,
        #         H_out=mask_rendered.shape[-2],
        #         W_out=mask_rendered.shape[-1],
        #     )
        #     if broadcast_batch_and_cams:
        #         object_mask_res = object_mask_res[:, None]
        #     object_mask_sim_pxl = (
        #         1.0 - ((object_mask_res - mask_rendered) ** 2)[..., 0, :, :]
        #     )
        #     if feats2d_img_mask is not None:
        #         if broadcast_batch_and_cams:
        #             object_mask_sim_pxl = (1.0 - 1.0 * feats2d_img_mask) + (
        #                 1.0 * feats2d_img_mask
        #             ) * object_mask_sim_pxl
        #         else:
        #             object_mask_sim_pxl = (
        #                 1.0 - 1.0 * feats2d_img_mask[..., 0, :, :]
        #             ) + (1.0 * feats2d_img_mask[..., 0, :, :]) * object_mask_sim_pxl
        #
        #     object_mask_sim_pxl = object_mask_sim_pxl * 1.0
        #     object_mask_sim = object_mask_sim_pxl.flatten(-2).mean(dim=-1)
        #     if return_sim_pxl:
        #         sim, sim_pxl = sim
        #         sim = sim + object_mask_sim
        #         sim_pxl = sim_pxl + object_mask_sim_pxl
        #         return sim, sim_pxl
        #     else:
        #         sim += object_mask_sim
        #         return sim

        return sim

    def sample_nearest_to_feats2d_img(
        self,
        feats2d_img,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        imgs_sizes=None,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=False,
        add_other_objects=False,
        dtype=None,
        device=None,
        smooth_labels=False,
        sim_temp=1.0,
        instance_deform=None,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True
        """

        sim_feats2d = self.get_sim_feats2d_img_to_all(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            dense=True,
            sim_temp=sim_temp,
            clutter_pxl2d=None,
            instance_deform=instance_deform,
        )

        label_feats2d_nearest = sim_feats2d.argmax(
            dim=1,
            keepdim=True,
        )  # (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True

        nearest_mods2d = self.sample(
            modalities=modalities,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            sample_clutter=add_clutter,
            sample_other_objects=add_other_objects,
            instance_deform=instance_deform,
        )  # BxV(+1)x3

        from od3d_basic.cv.select import batched_index_select

        B, _, H, W = label_feats2d_nearest.shape
        nearest_mods2d = (
            batched_index_select(
                input=nearest_mods2d,
                index=label_feats2d_nearest.flatten(1),
                dim=1,
            )
            .permute(0, 2, 1)
            .reshape(B, 3, H, W)
        )  # Bx3xHxW
        return nearest_mods2d

    def get_sim_feats2d_img_to_all(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj=None,
        cams_intr4x4=None,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        add_clutter=True,
        add_other_objects=True,
        coarse_labels=False,
        dense=False,
        sim_temp=1.0,
        clutter_pxl2d=None,
        return_feats=False,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True
        """

        if coarse_labels:
            feat_modality = PROJECT_MODALITIES.FEATS_COARSE
        else:
            feat_modality = PROJECT_MODALITIES.FEATS

        if dense:
            feats1d_sampled = self.sample(
                modalities=feat_modality,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                instance_deform=instance_deform,
                detach_objects=detach_objects,
                detach_deform=detach_deform,
            )
            sim_feats2d = self.get_sim_feats2d_img_and_feats1d_obj(
                feats2d_img,
                feats1d_sampled,
                add_clutter=False,
                temp=sim_temp,
            )
            if return_feats:
                return sim_feats2d, feats2d_img
            else:
                return sim_feats2d

        else:
            mods1d_sampled = self.sample_with_img2d(
                img2d=feats2d_img,
                modalities=[feat_modality, PROJECT_MODALITIES.IMG],
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                clutter_pxl2d=clutter_pxl2d,
                dtype=feats2d_img.dtype,
                device=feats2d_img.device,
                instance_deform=instance_deform,
                detach_objects=detach_objects,
                detach_deform=detach_deform,
            )

            sim_feats1d = self.get_sim_feats1d_img_and_feats1d_obj(
                mods1d_sampled[PROJECT_MODALITIES.IMG],
                mods1d_sampled[feat_modality],
                add_clutter=False,
                temp=sim_temp,
            )
            # sim_feats1d[feats1d_obj_mask[:, None,]] = -1

            if return_feats:
                return sim_feats1d, mods1d_sampled[PROJECT_MODALITIES.IMG]
            else:
                return sim_feats1d

    def get_label_feats2d_img(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        add_clutter=True,
        add_other_objects=True,
        sample_clutter_count=5,
        dense=False,
        smooth_labels=False,
        coarse_labels=False,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            label (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True or (B, V(+1),V+N) if smooth_labels=True
            label_mask (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True
            noise_pxl2d (torch.Tensor): (B, V, 2), or None if dense=True
        """

        if dense:
            if coarse_labels:
                label_modality = PROJECT_MODALITIES.ONEHOT_COARSE
            else:
                if smooth_labels:
                    label_modality = PROJECT_MODALITIES.ONEHOT_SMOOTH
                else:
                    label_modality = PROJECT_MODALITIES.ONEHOT

            label2d = self.render(
                modalities=label_modality,
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                instance_deform=instance_deform,
                detach_objects=detach_objects,
                detach_deform=detach_deform,
            )

            if add_clutter and feats2d_img_mask is not None:
                label2d_clutter = torch.zeros_like(label2d)
                label2d_clutter[:, -1] = 1.0
                feats2d_img_mask = 1.0 * (feats2d_img_mask.detach().clone() > 0.5)
                label2d = (
                    feats2d_img_mask * label2d
                    + (1.0 - feats2d_img_mask) * label2d_clutter
                )  # BxKxHxW

            return label2d, None, None

        else:
            mask_modality = PROJECT_MODALITIES.MASK
            if coarse_labels:
                label_modality = PROJECT_MODALITIES.ONEHOT_COARSE
            else:
                if smooth_labels:
                    label_modality = PROJECT_MODALITIES.ONEHOT_SMOOTH
                else:
                    label_modality = PROJECT_MODALITIES.ID

            mods1d_sampled = self.sample_with_img2d(
                img2d=feats2d_img,
                img2d_mask=feats2d_img_mask,
                modalities=[
                    mask_modality,
                    label_modality,
                    PROJECT_MODALITIES.CLUTTER_PXL2D,
                ],
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_sizes=imgs_sizes,
                objects_ids=objects_ids,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                down_sample_rate=down_sample_rate,
                add_clutter=add_clutter,
                add_other_objects=add_other_objects,
                sample_clutter_count=sample_clutter_count,
                instance_deform=instance_deform,
                detach_objects=detach_objects,
                detach_deform=detach_deform,
            )

            return (
                mods1d_sampled[label_modality],
                mods1d_sampled[mask_modality],
                mods1d_sampled[PROJECT_MODALITIES.CLUTTER_PXL2D],
            )

    def get_label_clutter(self, add_other_objects=False, one_hot=False, device=None):
        raise NotImplementedError

    def get_label_and_sim_feats2d_img_to_all(
        self,
        feats2d_img,
        imgs_sizes,
        cams_tform4x4_obj,
        cams_intr4x4,
        objects_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
        feats2d_img_mask=None,
        add_clutter=True,
        add_other_objects=True,
        sample_clutter_count=5,
        dense=False,
        smooth_labels=False,
        coarse_labels=False,
        sim_temp=1.0,
        return_feats=False,
        instance_deform=None,
        detach_objects=False,
        detach_deform=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxFxHxW
            feats1d_obj (torch.Tensor): BxVxF
        Returns:
            label (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True or (B, V(+1),V+N) if smooth_labels=True
            label_mask (torch.Tensor): (B, V+N) or (B, V(+1), H, W) if dense=True
            noise_pxl2d (torch.Tensor): (B, V, 2)
            sim_feats (torch.Tensor): (B, V(+1), V+N) or (B, V(+1), H, W) if dense=True

        """
        label_feats, label_feats_mask, feat_clutter_pxl2d = self.get_label_feats2d_img(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            feats2d_img_mask=feats2d_img_mask,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            sample_clutter_count=sample_clutter_count,
            dense=dense,
            smooth_labels=smooth_labels,
            coarse_labels=coarse_labels,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
        )

        sim_feats = self.get_sim_feats2d_img_to_all(
            feats2d_img=feats2d_img,
            imgs_sizes=imgs_sizes,
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            objects_ids=objects_ids,
            broadcast_batch_and_cams=broadcast_batch_and_cams,
            down_sample_rate=down_sample_rate,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            coarse_labels=coarse_labels,
            dense=dense,
            sim_temp=sim_temp,
            clutter_pxl2d=feat_clutter_pxl2d,
            return_feats=return_feats,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
        )

        if return_feats:
            return (
                label_feats,
                label_feats_mask,
                feat_clutter_pxl2d,
                sim_feats[0],
                sim_feats[1],
            )
        else:
            return label_feats, label_feats_mask, feat_clutter_pxl2d, sim_feats

    def update_feats_moving_average(
        self,
        labels,
        labels_mask,
        feats,
        alpha,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        raise NotImplementedError

    def update_feats_total_average(
        self,
        labels,
        labels_mask,
        feats,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        raise NotImplementedError

    def get_sim_cams_verts(
        self,
        categories_ids,
        feats2d_net,
        verts2d_mesh,
        verts2d_mesh_mask,
        return_sim_pxl=False,
        feats2d_net_mask=None,
    ):
        pass

    def render_with_clutter(self):
        pass

    def get_sim_feats1d_img_and_feats1d_obj(
        self,
        feats1d_img,
        feats1d_obj,
        add_clutter=False,
        temp=1.0,
    ):
        """
        Args:
            feats1d_img (torch.Tensor): BxNxC
            feats1d_obj (torch.Tensor): BxVxC

        Returns:
            sim_feats1d (torch.Tensor): BxVxN or BxV+1xN if add_clutter=True
        """
        if add_clutter:
            feats1d_obj = torch.cat([feats1d_obj, self.feat_clutter[None, None]], dim=1)

        sim_feats1d = self.get_sim(
            "bnc,bvc->bvn",
            feats1d_img,
            feats1d_obj,
            temp=temp,
        )
        return sim_feats1d

    def get_sim_feats2d_img_and_feats1d_obj(
        self,
        feats2d_img,
        feats1d_obj,
        add_clutter=False,
        temp=1.0,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats1d_obj (torch.Tensor): BxVxC

        Returns:
            sim_feats2d (torch.Tensor): BxVxHxW or BxV+1xHxW if add_clutter=True
        """
        if add_clutter:
            feats1d_obj = torch.cat([feats1d_obj, self.feat_clutter[None, None]], dim=1)

        sim_feats2d = self.get_sim(
            "bchw,bvc->bvhw",
            feats2d_img,
            feats1d_obj,
            temp=temp,
        )
        return sim_feats2d

    def get_sim_feats2d_img_and_rendered(
        self,
        feats2d_img,
        feats2d_rendered,
        return_sim_pxl=False,
        feats2d_img_mask=None,
        allow_clutter=True,
        temp=1.0,
        objects_ids=None,
        normalize_surface=False,
        use_neg_mse=False,
    ):
        """
        Args:
            feats2d_img (torch.Tensor): BxCxHxW
            feats2d_rendered (torch.Tensor): Bx(T)xCxHxW
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_img_mask (torch.Tensor): Bx1xHxW

        Returns:
            sim (torch.Tensor): Bx(T)
            sim_feats2d (torch.Tensor, optional): Bx(T)xHxW
        """
        if feats2d_rendered.dim() == 5:
            sim_feats2d = self.get_sim(
                "bchw,bvchw->bvhw",
                feats2d_img,
                feats2d_rendered,
                temp=temp,
                use_neg_mse=use_neg_mse,
            )
        else:
            sim_feats2d = self.get_sim(
                "bchw,bchw->bhw",
                feats2d_img,
                feats2d_rendered,
                temp=temp,
                use_neg_mse=use_neg_mse,
            )[:, None]

        # shape (B, V, H, W)
        sim_clutter2d = self.get_sim(
            "bchw,c->bhw",
            feats2d_img,
            self.feat_clutter,
            temp=temp,
            use_neg_mse=use_neg_mse,
        )[:, None].expand(sim_feats2d.shape)

        if feats2d_img_mask is not None:
            clutter_mask = (feats2d_img_mask < 0.5).expand(sim_feats2d.shape)
        else:
            clutter_mask = torch.zeros_like(sim_feats2d, dtype=torch.bool)

        if allow_clutter:
            clutter_mask = clutter_mask | (sim_feats2d < sim_clutter2d)

        sim_feats2d[clutter_mask] = sim_clutter2d[clutter_mask]

        if normalize_surface:
            # shape (B, V, H, W)
            sim_surface2d = self.get_sim(
                "bchw,bfc->bfhw",
                feats2d_img,
                self.sample(
                    modalities=PROJECT_MODALITIES.FEATS_COARSE,
                    objects_ids=objects_ids,
                    add_clutter=True,
                ),
                temp=temp,
                use_neg_mse=use_neg_mse,
            )
            sim_feats2d = torch.exp(sim_feats2d) / torch.exp(sim_surface2d).sum(
                dim=1,
                keepdim=True,
            )

        sim = sim_feats2d.flatten(2).mean(dim=-1)

        if feats2d_rendered.dim() == 4:
            sim_feats2d = sim_feats2d.squeeze(1)
            sim = sim.squeeze(1)

        if return_sim_pxl:
            return sim, sim_feats2d
        else:
            return sim

    def get_sim(self, comb, featsA, featsB, temp=1.0, use_neg_mse=False):
        """
        Args:
            comb (str): e.g. hwf,hwf->hw
            featsA (torch.Tensor): e.g. shape (H, W, F)
            featsB (torch.Tensor): e.g. shape (H, W, F)

        Returns:
            sim (torch.Tensor): e.g. shape (H, W)
        """
        if not use_neg_mse:
            if self.feats_distribution == FEATS_DISTR.VON_MISES_FISHER:
                return torch.einsum(comb, featsA, featsB) / temp
            elif self.feats_distribution == FEATS_DISTR.GAUSSIAN:
                from od3d_basic.cv.geometry.dist import einsum_cdist

                return -einsum_cdist(comb, featsA, featsB) / temp
            else:
                msg = f"Unknown distribution {self.feats_distribution}"
                raise NotImplementedError(msg)
        else:
            from od3d_basic.cv.geometry.dist import einsum_cdist

            return -einsum_cdist(comb, featsA, featsB)
