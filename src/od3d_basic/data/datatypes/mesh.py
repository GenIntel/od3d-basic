from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import Tensor
import torch


@dataclass
class Mesh:
    verts:       Tensor           # (V, 3)
    faces:       Tensor           # (F, 3)  int64
    vert_colors: Optional[Tensor] = None  # (V, 3) float RGB
    verts_uvs:   Optional[Tensor] = None  # (V, 2) float UV — y-flipped to image convention
    faces_uvs:   Optional[Tensor] = None  # (F, 3) int64 UV face indices
    texture:     Optional[Tensor] = None  # (3, H, W) float


def convert_mesh(type: str, mesh: Mesh) -> Mesh:
    # if type has format mc64, convert to marching cubes mesh
    if type.startswith("mc"):
        res = int(type[2:])
        return convert_mesh_to_mc(mesh, res)
    else:
        raise ValueError(f"Unknown mesh type: {type}")
    
def convert_mesh_to_mc(mesh: Mesh, res: int = 64):
    import igl
    import numpy as np
    V, F = mesh.verts.cpu().numpy(), mesh.faces.cpu().numpy()

    # -----------------------------
    # 2. Create volumetric grid
    # -----------------------------
    # Bounding box
    min_v = V.min(axis=0)
    max_v = V.max(axis=0)
    # Add padding
    padding = 0.05 * (max_v - min_v)
    min_v -= padding
    max_v += padding
    # Grid resolution (increase for higher quality)
    res = 64
    x = np.linspace(min_v[0], max_v[0], res)
    y = np.linspace(min_v[1], max_v[1], res)
    z = np.linspace(min_v[2], max_v[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # -----------------------------
    # 3. Compute signed distance
    # -----------------------------
    # Returns signed distances to mesh
    S, I, C, _ = igl.signed_distance(grid_points, V, F)

    # Reshape to 3D grid
    SDF = S.reshape((res, res, res))
    # -----------------------------
    # 4. Extract isosurface (0 level set)
    # -----------------------------
    verts, faces, _ = igl.marching_cubes(
        SDF.reshape(-1),
        grid_points, # .reshape((res, res, res) + (3,)),
        res, # x,
        res, # y,
        res, # z,
        0.0  # iso-value
    )
    
    print("Watertight mesh:", verts.shape[0], "vertices,", faces.shape[0], "faces")

    mesh = Mesh(
        verts=torch.from_numpy(verts).float(),
        faces=torch.from_numpy(faces).long()
    )
    
    return mesh
