from od3d_basic.cv.transforms.transform import OD3D_Transform
from od3d_basic.od3d_datasets.frame import OD3D_FRAME_MODALITIES
from torchvision.transforms.transforms import Normalize
import torch

class RGB_Normalize(OD3D_Transform):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, frame):
        if OD3D_FRAME_MODALITIES.RGB in frame.modalities:
            if frame.rgb.dim() == 4:
                frame.rgb = torch.stack([self.normalize(_rgb) for _rgb in frame.get_rgb()], dim=0)
            else:
                frame.rgb = self.normalize(frame.get_rgb())
        if OD3D_FRAME_MODALITIES.RGBS in frame.modalities:
            frame.rgbs = [self.normalize(rgb) for rgb in frame.get_rgbs()]
        return frame
