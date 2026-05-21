import logging

logger = logging.getLogger(__name__)
from od3d_basic.cv.geometry.objects3d.dmtet_x_gaussians.dmtet_x_gaussians import (
    DMTet_x_Gaussians,
)
from od3d_basic.cv.geometry.objects3d.meshes.meshes import Meshes, OD3D_Meshes_Deform
from od3d_basic.cv.geometry.objects3d.meshes_x_gaussians import Meshes_x_Gaussians
from typing import Union, List
from od3d_basic.cv.geometry.objects3d.objects3d import PROJECT_MODALITIES


class DMTet(DMTet_x_Gaussians):
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
        detach_objects=False,
        detach_deform=False,
        **kwargs,
    ):
        return super(Meshes_x_Gaussians, self).render_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            **kwargs,
        )

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
        instance_deform: OD3D_Meshes_Deform = None,
        detach_objects=False,
        detach_deform=False,
        **kwargs,
    ):
        return super(Meshes_x_Gaussians, self).sample_batch(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            objects_ids=objects_ids,
            modalities=modalities,
            add_clutter=add_clutter,
            add_other_objects=add_other_objects,
            device=device,
            dtype=dtype,
            sample_clutter=sample_clutter,
            sample_other_objects=sample_other_objects,
            instance_deform=instance_deform,
            detach_objects=detach_objects,
            detach_deform=detach_deform,
            **kwargs,
        )
