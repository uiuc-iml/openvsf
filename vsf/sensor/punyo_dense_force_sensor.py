from typing import Union
import numpy as np
import torch
import trimesh
from klampt import RobotModel, RobotModelLink, RigidObjectModel
from ..utils.data_utils import transform_points,transform_directions
from .base_sensor import BaseSensor, SimState
from typing import Union,Tuple,Dict


def barycentric_weights(vertices, triangles, points):
    """
    A utility function to compute barycentric weights for a set of points on a triangle mesh.
    
    Args:
        vertices (np.ndarray): The vertices of the mesh, shape (n_vertices, 3).
        triangles (np.ndarray): The triangles of the mesh, shape (n_triangles, 3).
        points (np.ndarray): The points for which to compute barycentric weights, shape (n_points, 3).
    Returns:
        weight (np.ndarray): The barycentric weights for each point, shape (n_points, n_vertices).
    """
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
    
    This module simulates dense contact forces at every vertex of a triangle mesh. 
    Each sensor reading is a (3 * num_vertices)-dimensional vector, where entries
    (3*i, 3*i+1, 3*i+2) represent the local x, y, z force at the i-th vertex in the mesh.

    SimState contains contact forces in world coordinates; the sensor internally converts them
    into local coordinates. For a single body, the sensor's Jacobian matrix—which maps
    world-space contact forces to local vertex forces—has dimensions: (3 * num_vertices, num_contact_points, 3).
    Multiple bodies can be processed by summing their individual contributions.

    Two operational modes are supported:

    1) **Per-Vertex Mode**  
    In this simpler mode, each contact point directly corresponds to a single mesh
    vertex. The local force at vertex i is the sum of all contact forces (transformed
    to local coordinates) that map to i:

    .. math::
        \\mathbf{F}_{i} = \\sum_{c \\in C_i} (\\mathbf{R}_{\\text{local}} \\; \\mathbf{f}_c^w),

    where :math:`\\mathbf{f}_c^w` is the c-th contact force in world coordinates,
    :math:`\\mathbf{R}_{\\text{local}}` is the 3x3 rotation from world frame to the
    sensor's local frame, and :math:`C_i` is the set of contact points mapped to vertex i.

    - **Jacobian Construction**:  
        The Jacobian block for vertex i and contact c is :math:`\\mathbf{R}_{\\text{local}}`
        if contact c is associated with i; otherwise, it is zero. Symbolically:

        .. math::
            J[3i : 3(i+1),\\; c,\\; :] = \\mathbf{R}_{\\text{local}}.

    2) **Barycentric Interpolation Mode**  
    In this mode, a contact point c may lie on a triangular (or polygonal) face of the
    mesh rather than a single vertex. We compute barycentric weights :math:`w_{i,c}`
    so that the force is split among the face's vertices. The local force at vertex i is:

    .. math::
        \\mathbf{F}_{i} = \\sum_{c} w_{i,c} \\bigl(\\mathbf{R}_{\\text{local}} \\; \\mathbf{f}_c^w\\bigr),

    where :math:`w_{i,c}` is the barycentric weight of vertex i for contact c.

    - **Jacobian Construction**:  
        The block for vertex i, contact c is :math:`w_{i,c} \\, \\mathbf{R}_{\\text{local}}`.
        Thus:

        .. math::
            J[3i : 3(i+1),\\; c,\\; :] = w_{i,c} \\, \\mathbf{R}_{\\text{local}}.

    In both modes, the sensor aggregates these transformed forces into a single output
    vector of shape (3 * num_vertices), storing local x, y, z forces per vertex in order.
    The returned Jacobian is organized as a dense tensor of shape (3 * num_vertices, num_contact_points, 3).

        
    Attributes:
        name (str): The name of the sensor.
        attachModelName (str): The name of the robot link or rigid object that the sensor is attached to.
        vertices (np.ndarray): The vertices of the mesh.
        triangles (np.ndarray): The triangles of the mesh.

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
        """Returns 3*num_vertices vector of forces.  Preserves torch gradients"""
        return self.predict_torch(state)
    
    def predict_torch(self, state:SimState) -> torch.Tensor:
        """Return dense contact forces as a 3*num_vertices vector."""
        Hlocal = torch.linalg.inv(state.body_transforms[self.attachModelName])
        Rlocal = Hlocal[:3,:3]
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        
        tsr_params = {'device': state.device, 'dtype': state.dtype}
        
        forces = torch.zeros((len(self.vertices),3), **tsr_params)
        for body,contact_state in zip(bodies,contact_states):
            if contact_state.elems1 is not None and len(contact_state.elems1) > 0:
                assert len(contact_state.elems1) == len(contact_state.points)
                forces[contact_state.elems1, :] += transform_directions(contact_state.forces, Rlocal)
            elif contact_state.elems2 is not None and len(contact_state.elems2) > 0:
                assert len(contact_state.elems2) == len(contact_state.points)
                # transform points to local coordinates
                points_local = transform_points(contact_state.points, Hlocal)
                # barycentric weights
                weights = barycentric_weights(self.vertices, self.triangles, points_local.cpu().numpy())
                weights = torch.from_numpy(weights).to(**tsr_params)
                # transform forces to local coordinates
                forces_local = transform_directions(contact_state.forces, Rlocal)
                forces += weights.transpose(0, 1) @ forces_local
            else:
                raise ValueError("No contact points found on the mesh")

        return forces.reshape(-1)
        
    def measurement_force_jacobian(self, state: SimState) -> Dict[Tuple[str,str],torch.Tensor]:
        """Result should have shape (3*num_vertices,num_contact_points,3)"""
        Hlocal = torch.linalg.inv(state.body_transforms[self.attachModelName])
        Rlocal = Hlocal[:3,:3]
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        
        tsr_params = {'device': state.device, 'dtype': state.dtype}
        
        jacs = {}
        for body,contact_state in zip(bodies,contact_states):
            J = torch.zeros((len(self.vertices)*3, len(contact_state.points), 3), **tsr_params)
            if contact_state.elems1 is not None and len(contact_state.elems1) > 0:
                for c,i in enumerate(contact_state.elems1):
                    J[3*i:3*(i+1),c,:] = Rlocal
            elif contact_state.elems2 is not None and len(contact_state.elems2) > 0:
                # transform points to local coordinates
                points_local = transform_points(contact_state.points, Hlocal)
                # barycentric weights
                weights = barycentric_weights(self.vertices, self.triangles, points_local.cpu().numpy())
                weights = torch.from_numpy(weights).to(**tsr_params)
                for d in range(3):
                    for e in range(3):
                        # J[d::3, :, e] corresponds to all rows in the flattened dimension
                        # that pick out dimension "d" of each vertex
                        # and partial derivative w.r.t. dimension "e" in the last axis
                        J[d::3, :, e] = Rlocal[d, e] * weights.T
            jacs[self.attachModelName,body] = J
        return jacs