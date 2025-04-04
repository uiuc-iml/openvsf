import meshio
import numpy as np

def unpack_vtk_mesh(mesh_fn):
    '''
    Unpack .vtk mesh file to points, triangles, boundary, boundary_mask
    
    Args:
    - mesh_fn: str, path to the .vtk mesh file
    
    Returns a tuple of:
    - points: np.ndarray, shape [N, 3], vertices of the mesh
    - triangles: np.ndarray, shape [N, 3], triangles of the mesh
    - boundary: np.ndarray, shape [N], boundary of the mesh
    - boundary_mask: np.ndarray, shape [N], mask of the boundary
    '''
    mesh = meshio.read(mesh_fn)
    points = mesh.points.astype(np.float32)
    triangles = np.array(mesh.cells_dict['triangle'], dtype=np.int32)
    boundary = np.array(mesh.cells_dict['vertex'], dtype=np.int32)
    boundary_mask = np.zeros(len(points), dtype=np.uint8)
    for n in boundary:
        boundary_mask[n] = 1
    return points, triangles, boundary, boundary_mask
