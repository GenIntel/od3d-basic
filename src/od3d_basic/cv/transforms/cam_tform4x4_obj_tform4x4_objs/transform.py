import torchvision
from od3d_basic.cv.geometry.transform import get_scale1d_tform4x4
from od3d_basic.cv.geometry.transform import scale1d_tform4x4_with_dist
from od3d_basic.cv.transforms.transform import OD3D_Transform

import time

# Unroll Objs in Image into Batch. 
class CamTform4x4ObjTform4x4Objs(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        # start_time = time.time()
        if "cam_tform4x4_obj" in frame.modalities:
            frame.obj_tform4x4_objs = frame.get_obj_tform4x4_objs()[:]
            obj_tform4x4_objs = frame.obj_tform4x4_objs[0] # frame.get_obj_tform4x4_objs()[0]
            cam_tform4x4_obj = frame.get_cam_tform4x4_obj()
            cam_tform4x4_obj = cam_tform4x4_obj @ obj_tform4x4_objs
            frame.cam_tform4x4_obj = scale1d_tform4x4_with_dist(cam_tform4x4_obj) # 0.03 of 0.12
        if "obj_syms" in frame.modalities:
            frame.obj_syms = frame.get_objs_syms()[0]
        if "obj_kpts3d" in frame.modalities:
            frame.obj_kpts3d = frame.get_objs_kpts3d()[0]
        if "obj_kpts3d_mask" in frame.modalities:
            frame.obj_kpts3d_mask = frame.get_objs_kpts3d_mask()[0]
        if "obj_kpts2d_mask" in frame.modalities:
            frame.obj_kpts2d_mask = frame.get_objs_kpts2d_mask()[0]
        if "bbox" in frame.modalities:
            bboxs = frame.get_bboxs()
            if bboxs.dim() == 1:
                frame.bbox = bboxs
            else:
                frame.bbox = bboxs[0]


        # # 0.05 of 0.12
        if "rgb" in frame.modalities:
            frame.rgb_orig = frame.get_rgb_orig()

            #print(f"transform takes {round(time.time() - start_time, 2)} seconds")

        if hasattr(frame, 'pair') and frame.pair is not None:
            frame.pair = self(frame.pair)
        return frame
