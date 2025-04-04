from typing import Union
import numpy as np
import torch
import trimesh
from klampt import RobotModel, RobotModelLink, RigidObjectModel
from ..utils.data_utils import transform_points,transform_directions
from .base_sensor import BaseSensor, SimState
from typing import Union,Tuple,Dict


def barycentric_weights(vertices, triangles, points):
    """A utility function to compute barycentric weights for a set of points on a mesh."""
    tri_mesh = trimesh.Trimesh(vertices, triangles)
    # find closest triangle
    closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(tri_mesh, points)
    triangles_nearest = tri_mesh.triangles[closest_tri_idx]
    barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, points)

    weight = np.zeros((points.shape[0], vertices.shape[0]))
    point_idx = np.arange(points.shape[0])
    for pt_idx, bary, tri_idx in zip(point_idx, barycentric, closest_tri_idx):
        tri = triangles[tri_idx]
        for i, tri_v in enumerate(tri):
            weight[pt_idx, tri_v] = bary[i]
    return weight

class PunyoDenseForceSensor(BaseSensor):
    """A sensor model for the Punyo sensor providing dense force estimates
    on each vertex of the model's mesh.
        
    Attributes:
        name (str): The name of the sensor.
        attachModelName (str): The name of the robot link or rigid object that the sensor is attached to.
        vertices (np.ndarray): The vertices of the mesh.
        triangles (np.ndarray): The triangles of the mesh.
        boundary (np.ndarray): The boundary of the mesh.
        boundary_mask (np.ndarray): The boundary mask of the mesh.

    """
    def __init__(self, sensor_name:str, attachModelName: str):
        self.name = sensor_name
        self.attachModelName = attachModelName

    def attach(self, model: Union[RobotModelLink, RigidObjectModel]):
        self.object = model

        mesh = model.geometry().getTriangleMesh()
        self.vertices = mesh.getVertices()
        self.triangles = mesh.getIndices()
    
    def measurement_names(self):
        return sum([['p{}_x', 'p{}_y', 'p{}_z'] for i in range(len(self.vertices))],[])

    def predict(self, state: SimState) -> torch.Tensor:
        """Returns 3N vector of forces.  Preserves torch gradients"""
        return self.predict_torch(state)
    
    def predict_torch(self, state:SimState) -> torch.Tensor:
        """Return dense contact forces as a 3N vector."""
        Rlocal = state.body_transforms[self.attachModelName][:3,:3].T
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        forces = torch.zeros((len(self.vertices),3),device=state.device)
        for body,contact_state in zip(bodies,contact_states):
            assert len(contact_state.elems1) > 0,'TODO: dense force estimation for non-mesh-aligned points?'
            assert len(contact_state.elems1) == len(contact_state.points)
            # temproary fix, ideally device should propagate through the whole simulation
            forces[contact_state.elems1] += transform_directions(contact_state.forces, Rlocal).to(state.device)
        return forces.reshape(-1)
        
    def measurement_force_jacobian(self, state: SimState) -> Dict[Tuple[str,str],torch.Tensor]:
        """Result should have shape (num_vertices*3,num_vertices,3)"""
        Rlocal = state.body_transforms[self.attachModelName][:3,:3].T
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        jacs = {}
        for body,contact_state in zip(bodies,contact_states):
            assert len(contact_state.elems1) > 0,'TODO: dense force estimation for non-mesh-aligned points?'
            J = torch.zeros((len(contact_state.points),3,len(self.vertices)*3))
            for c,i in enumerate(contact_state.elems1):
                J[3*i:3*(i+1),c,:] = Rlocal
            jacs[self.attachModelName,body] = J
        return jacs