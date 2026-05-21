import torchvision
from od3d_basic.cv.transforms.transform import OD3D_Transform
import torch

class RGB_Random(OD3D_Transform):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomApply(
                #    torchvision.transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)), p=0.1),
                torchvision.transforms.RandomSolarize(threshold=128, p=0.2),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        ),
                    ],
                    p=0.8,
                ),
            ],
        )

    def __call__(self, frame):
        rgb = frame.get_rgb()
        if rgb.dim() == 4:
            rgb = torch.stack([self.transform(_rgb) for _rgb in rgb], dim=0)
        else:
            rgb = self.transform(rgb)
        frame.rgb = rgb
        return frame
