import klampt
import numpy as np

def load_trimesh_preserve_vertices(fn : str) -> klampt.Geometry3D:
    """Loads a klampt Geometry3D from a triangle mesh file, ensuring
    that the vertices are preserved exactly.  Klamp't's default
    [loader may reorder vertices](https://github.com/krishauser/Klampt/issues/205),
    which is not desirable for controlled deformable meshes.
    """
    import trimesh
    mesh_trimesh = trimesh.load_mesh(fn)
    mesh = klampt.TriangleMesh()
    mesh.setVertices(mesh_trimesh.vertices)
    mesh.setIndices(mesh_trimesh.faces.astype(np.int32))
    return klampt.Geometry3D(mesh)
