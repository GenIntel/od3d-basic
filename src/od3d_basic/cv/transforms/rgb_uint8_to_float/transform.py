import logging

logger = logging.getLogger(__name__)
from od3d_basic.cv.transforms.transform import OD3D_Transform
from od3d_basic.od3d_datasets.frame import OD3D_FRAME_MODALITIES


class RGB_UInt8ToFloat(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        if OD3D_FRAME_MODALITIES.RGB in frame.modalities:
            frame.rgb = frame.get_rgb() / 255.0
        if OD3D_FRAME_MODALITIES.RGBS in frame.modalities:
            frame.rgbs = [rgb / 255.0 for rgb in frame.get_rgbs()]
        return frame
