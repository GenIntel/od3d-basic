import torch
from od3d_basic.cv.geometry.transform import scale1d_tform4x4_with_dist, proj3d2d_broadcast
from od3d_basic.cv.transforms.transform import OD3D_Transform
from od3d_basic.od3d_datasets.frame import OD3D_FRAME_MODALITIES
from copy import deepcopy

class PairUnfold(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        frame.modalities = deepcopy(frame.modalities)
        if hasattr(frame, 'pair') and frame.pair is not None:
            frame.rgbs = torch.stack([frame.get_rgb(), frame.pair.get_rgb()], dim=0)
            if OD3D_FRAME_MODALITIES.RGBS not in frame.modalities:
                frame.modalities.append(OD3D_FRAME_MODALITIES.RGBS) 
            
            if OD3D_FRAME_MODALITIES.RGB_ORIG in frame.modalities: # hasattr(frame, 'rgb_orig') and frame.rgb_orig is not None:
                frame.rgbs_orig = torch.stack([frame.get_rgb_orig(), frame.pair.get_rgb_orig()], dim=0)
                if OD3D_FRAME_MODALITIES.RGBS_ORIG not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.RGBS_ORIG) 
            
            if OD3D_FRAME_MODALITIES.CAM_INTR4X4 in frame.modalities:
                frame.cam_intr4x4s = torch.stack([frame.get_cam_intr4x4(), frame.pair.get_cam_intr4x4()], dim=0)
                if OD3D_FRAME_MODALITIES.CAM_INTR4X4S not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.CAM_INTR4X4S)

            if OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ in frame.modalities:
                frame.cam_tform4x4_objs = torch.stack([frame.get_cam_tform4x4_obj(), frame.pair.get_cam_tform4x4_obj()], dim=0)
                if OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJS not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJS)

            if OD3D_FRAME_MODALITIES.OBJ_KPTS3D in frame.modalities:
                frame_ref_kpts2d = proj3d2d_broadcast(proj4x4=frame.get_cam_proj4x4_obj(),
                                                    pts3d=frame.get_obj_kpts3d())
                frame_ref_kpts2d_mask = frame.get_obj_kpts2d_mask()
                
                frame_src_kpts2d = proj3d2d_broadcast(proj4x4=frame.pair.get_cam_proj4x4_obj(),
                                                    pts3d=frame.pair.get_obj_kpts3d())
                frame_src_kpts2d_mask = frame.pair.get_obj_kpts2d_mask()

                frame_kpts2d_mask = frame_src_kpts2d_mask * frame_ref_kpts2d_mask
                frame.kpts2d_annots = torch.stack([frame_ref_kpts2d[frame_kpts2d_mask], frame_src_kpts2d[frame_kpts2d_mask]], dim=0)

                frame.objs_kpts3d = torch.stack([frame.get_obj_kpts3d(), frame.pair.get_obj_kpts3d()], dim=0)
                frame.objs_kpts3d_mask = torch.stack([frame.get_obj_kpts3d_mask(), frame.pair.get_obj_kpts3d_mask()], dim=0)
                frame.objs_kpts2d_mask = torch.stack([frame_ref_kpts2d_mask, frame_src_kpts2d_mask], dim=0)
                
                if OD3D_FRAME_MODALITIES.OBJS_KPTS3D not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.OBJS_KPTS3D) 
                if OD3D_FRAME_MODALITIES.OBJS_KPTS3D_MASK not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.OBJS_KPTS3D_MASK)
                if OD3D_FRAME_MODALITIES.OBJS_KPTS2D_MASK not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.OBJS_KPTS2D_MASK) 
                if OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS) 

            if OD3D_FRAME_MODALITIES.SIZE in frame.modalities:
                frame.sizes = torch.stack([frame.size, frame.pair.size], dim=0)
                if OD3D_FRAME_MODALITIES.SIZES not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.SIZES) 

            if OD3D_FRAME_MODALITIES.BBOX in frame.modalities:
                frame_ref_bbox = frame.get_bbox()
                frame_src_bbox = frame.pair.get_bbox()
                frame.bboxs = torch.stack([frame_ref_bbox, frame_src_bbox], dim=0)
                if OD3D_FRAME_MODALITIES.BBOXS not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.BBOXS) 
            
            if OD3D_FRAME_MODALITIES.DEPTH in frame.modalities:
                frame.depths = torch.stack([frame.get_depth(), frame.pair.get_depth()], dim=0)
                if OD3D_FRAME_MODALITIES.DEPTHS not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.DEPTHS) 

            if OD3D_FRAME_MODALITIES.DEPTH_MASK in frame.modalities:
                frame.depths_masks = torch.stack([frame.get_depth_mask(), frame.pair.get_depth_mask()], dim=0)
                if OD3D_FRAME_MODALITIES.DEPTHS_MASKS not in frame.modalities:
                    frame.modalities.append(OD3D_FRAME_MODALITIES.DEPTHS_MASKS) 
            


            # frame.name_unique = f"{frame.name_unique}+{frame.pair.name_unique}"
            #frame.modalities = [mod for mod in deepcopy(frame.modalities) if mod is not None]
            #frame.modalities.append(OD3D_FRAME_MODALITIES.RGBS) # "rgbs"
            #frame.modalities.append(OD3D_FRAME_MODALITIES.KPTS2D_ANNOTS) # "kpts2d_annots"
            #frame.modalities.append(OD3D_FRAME_MODALITIES.BBOXS) # "bboxs"

            # rgb, mask, ..., cam_tform4x4_obj, cam_intr4x4,
            # category should remain for each pair, #batchwise

        return frame
