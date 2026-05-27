from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class Mesh:
    verts:       Tensor           # (V, 3)
    faces:       Tensor           # (F, 3)  int64
    vert_colors: Optional[Tensor] = None  # (V, 3) float RGB
    verts_uvs:   Optional[Tensor] = None  # (V, 2) float UV — y-flipped to image convention
    faces_uvs:   Optional[Tensor] = None  # (F, 3) int64 UV face indices
    texture:     Optional[Tensor] = None  # (3, H, W) float
