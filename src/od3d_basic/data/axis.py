from od3d_basic.data.ext_enum import StrEnum
import torch


class AXIS(StrEnum):
    BACK = "back"
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    NONE = "none"
    UNKNOWN = "unknown"

DEFAULT_AXIS = [AXIS.LEFT, AXIS.BACK, AXIS.TOP]

DEFAULT_AXIS_MAP = {
    AXIS.BACK: [0, 1, 0],
    AXIS.FRONT: [0, -1, 0],
    AXIS.LEFT: [1, 0, 0],
    AXIS.RIGHT: [-1, 0, 0],
    AXIS.TOP: [0, 0, 1],
    AXIS.BOTTOM: [0, 0, -1],
    AXIS.NONE: [0, 0, 0],
    AXIS.UNKNOWN: [0, 0, 0]
}

def get_default_rot3x3_from_axis(axis):
    # axis: List[AXIS] list should be of length 3
    if axis is None:
        return torch.eye(3)
    
    if len(axis) != 3:
        raise ValueError(f"Axis should be of length 3, but got {len(axis)}")
    
    default_rot3x3 = []
    for i, a in enumerate(axis):
        if a in DEFAULT_AXIS_MAP:
            default_rot3x3.append(DEFAULT_AXIS_MAP[a])
        else:
            default_rot3x3.append([0, 0, 0])

    default_rot3x3 = torch.tensor(default_rot3x3, dtype=torch.float32)

    # if three rows are zero then set to identity
    if default_rot3x3.abs().sum() == 0:
        return torch.eye(3)

        # if two rows are set one to the default axis iterating from last row
    if default_rot3x3.abs().sum() == 1:
        nonzero_row_id = None
        for i in range(3):                
            if default_rot3x3[i].abs().sum() == 1:
                nonzero_row_id = i
        
        for i in range(2, -1, -1):
            _default_axis = torch.tensor(DEFAULT_AXIS_MAP[DEFAULT_AXIS[i]], dtype=torch.float32)
            if (default_rot3x3[i].abs().sum() == 0) and ((default_rot3x3[nonzero_row_id].abs() - _default_axis.abs()).abs().sum() != 0):
                default_rot3x3[i] = _default_axis
                break

    # if one row is zero ensure that det = 1 and set that row to be the cross product of the other two rows
    if default_rot3x3.abs().sum() == 2:
        for i in range(3):
            if (default_rot3x3[i].abs().sum() == 0):
                default_rot3x3[i] = torch.cross(default_rot3x3[(i+1)%3].clone(), default_rot3x3[(i+2)%3].clone(), dim=-1)

    default_rot3x3 = torch.tensor(default_rot3x3, dtype=torch.float32).T
    
    if torch.det(default_rot3x3) < (1- 1e-8) or torch.det(default_rot3x3) > (1 + 1e-8):
        raise ValueError(f"Default rotation matrix should have determinant 1, but got {torch.det(default_rot3x3)} for axis {axis} with default_rot3x3 {default_rot3x3}")
    return default_rot3x3

def get_default_rot3x3_from_axes(axes):
    # axes: List[List[AXIS]] or None
    if axes is None:
        return torch.eye(3)

    if not isinstance(axes, list):
        raise ValueError(f"Axes should be a list of axis, but got {type(axes)}")
    # if list of list then recursively call, if list of AXIS then call get_default_rot3x3_from_axis
    if len(axes) == 0:
        return []
    #if isinstance(axes[0], list):

    return torch.stack([get_default_rot3x3_from_axis(axis) for axis in axes], dim=0)
    
    #else:        
    #    return get_default_rot3x3_from_axis(axes)  


def get_default_tform4x4_axisA(axisA):
    # axisA, axisB: List, List[AXIS]
    # returns tform4x4 that transforms from axisA to axisB
    from od3d_basic.cv.geometry.transform import transf4x4_from_rot3x3
    default_rot3x3_axisA = get_default_rot3x3_from_axes(axisA)
    return transf4x4_from_rot3x3(default_rot3x3_axisA)

def get_axisA_tform4x4_axisB(axisA, axisB):
    # axisA, axisB: List, List[AXIS]
    # returns tform4x4 that transforms from axisA to axisB
    from od3d_basic.cv.geometry.transform import transf4x4_from_rot3x3, inv_rot3x3, rot3x3

    default_rot3x3_axisA = get_default_rot3x3_from_axes(axisA)
    default_rot3x3_axisB = get_default_rot3x3_from_axes(axisB)

    axisA_rot3x3_axisB = rot3x3(inv_rot3x3(default_rot3x3_axisA), default_rot3x3_axisB)

    return transf4x4_from_rot3x3(axisA_rot3x3_axisB)

