from od3d_basic.cv.transforms.transform import OD3D_Transform


class Identity(OD3D_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        return frame
