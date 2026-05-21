import numpy as np
import trimesh
from trimesh.triangles import points_to_barycentric


def build_uv_mesh(uvs, uv_idx, verts, faces):
    """
    Create a trimesh mesh in 2D UV space with UV triangles corresponding
    to 3D triangles.
    """
    # Treat UVs as 3D points with z=0
    uv_verts_3d = np.hstack([uvs, np.zeros((uvs.shape[0], 1))])
    uv_mesh = trimesh.Trimesh(vertices=uv_verts_3d, faces=uv_idx, process=False)
    uv_mesh.visual = None  # Save memory
    return uv_mesh


def uv_to_3d(uv_points, uv_mesh, uvs, uv_idx, verts, faces):
    """
    Map 2D UV points to 3D coordinates on the mesh using barycentric interpolation.
    """
    # Convert to 3D (z=0)

    tids = trimesh.proximity.closest_point_uv(mesh, uv)
    barys = trimesh.triangles.uv_to_barycentric(mesh.visual.uv[mesh.faces[tids]], uv)
    points = np.einsum('ij,ijk->ik', barys, mesh.vertices[mesh.faces[tids]])

    trimesh.triangles.points_to_barycentric(triangles, points, method='cramer')

    uv_points_3d = np.hstack([uv_points, np.zeros((len(uv_points), 1))])

    # Find which UV triangle each point lies in

    closest_points, distances, face_idx = uv_mesh.nearest.on_surface(uv_points_3d)
    # face_idx, bary =

    # Initialize results
    points_3d = np.zeros((len(uv_points), 3))

    for i, (f, bary_coords) in enumerate(zip(face_idx, bary)):
        if f == -1:  # no hit
            points_3d[i] = np.nan
            continue

        # Find the corresponding 3D triangle
        v_idx = faces[f]
        tri_verts = verts[v_idx]

        # Barycentric interpolation in 3D
        points_3d[i] = (tri_verts * bary_coords[:, None]).sum(axis=0)

    return points_3d