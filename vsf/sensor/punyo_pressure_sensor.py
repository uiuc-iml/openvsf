# Base class for sensor model
import torch
import numpy as np
from klampt import RobotModel, RobotModelLink, RigidObjectModel
from klampt.model.geometry import vertex_normals
from .base_sensor import BaseSensor, SimState
from ..utils.data_utils import transform_directions, transform_points
import trimesh
from typing import Union, Dict, Tuple

def compute_vertex_areas(vertices : np.ndarray, triangles : np.ndarray) -> np.ndarray:
    """Computes the areas of triangles associated with each vertex, 
    i.e., one third of each triangle's area is added to each of its vertices."""
    vertex_areas = np.zeros(vertices.shape[0])
    for tri in triangles:
        v0,v1,v2 = vertices[tri]
        area = 0.5*np.linalg.norm(np.cross(v1-v0,v2-v0))
        vertex_areas[tri] += area / 3.0
    return vertex_areas

class PunyoPressureSensor(BaseSensor):
    r"""
    Punyo Pressure Sensor Model.

    This model produces a one-dimensional observation corresponding to a single
    pressure reading. The sensor is attached to a triangle mesh representation
    of the Punyo model, which can be regarded as a thin membrane or shell with a
    base area to which the mesh is fixed.

    Observation Model
    -----------------
    The observation model is derived from the force equilibrium equation on the
    membrane surface. When there is no external contact, the equilibrium can be
    expressed as:

    .. math::

       F_{tension} + \iint_{S} p \, d\mathbf{A} = 0,

    where :math:`p` is the (constant) pressure acting on the membrane, and the
    integral is taken over the closed surface :math:`S`. This integral simplifies
    because the net flux over a closed surface is equivalent to:

    .. math::

       \iint_{S} p \, d\mathbf{A} = p \, A_{base} \, \mathbf{n}_{base},

    where :math:`A_{base}` is the area of the base of the membrane, and
    :math:`\mathbf{n}_{base}` is its unit normal vector.

    After contact occurs, the force equilibrium equation is:

    .. math::

       F_{tension} + p^{\prime} \, A_{base} \, \mathbf{n}_{base} 
       + \iint_{S} \mathbf{F}_{vsf} \, dA = 0,

    where :math:`p^{\prime}` is the new pressure, and :math:`\mathbf{F}_{vsf}`
    represents the contact forces distributed over the surface. Assuming the
    change in membrane tension is relatively small, we focus on:

    .. math::

       (p^{\prime} - p) \, A_{base} \, \mathbf{n}_{base}
       = \iint_{S} \mathbf{F}_{vsf} \, dA.

    Taking the dot product with the unit normal :math:`\mathbf{n}_{base}`
    yields:

    .. math::

       (p^{\prime} - p) \, A_{base}
       = \iint_{S} \\bigl(\mathbf{F}_{vsf} \cdot \mathbf{n}_{base}\\bigr) \, dA.

    Consequently,

    .. math::

       (p^{\prime} - p) = \\frac{1}{A_{base}}
       \iint_{S} \\bigl(\mathbf{F}_{vsf} \cdot \mathbf{n}_{base}\\bigr) \, dA.

    In practice, this relationship gives a simple way to compute the pressure
    deviation based on the integrated contact forces normal to the membrane.

    Attributes
    ----------
    name : str
        The name of the sensor.
    attachModelName : str
        The name of the Punyo model to which the sensor is attached.
    object : RigidObjectModel
        The Punyo model object associated with this sensor.
    vertices : np.ndarray
        The array of vertex coordinates for the Punyo model, of shape (N, 3).
    triangles : np.ndarray
        The array of triangle indices for the Punyo model, of shape (M, 3).
    vertex_normals : torch.Tensor
        The normals at each vertex, of shape (N, 3).
    vertex_areas : torch.Tensor
        The area corresponding to each vertex (e.g., the area of its Voronoi
        region on the mesh), of shape (N, 1).
    scaled_vertex_normals : torch.Tensor
        The vertex normals scaled by their corresponding area, of shape (N, 3).

    """
    def __init__(self, name:str, punyo_link : str, 
                 base_normal_local: list = [0.0,0.0,1.0]):
        self.name: str = name
        self.attachModelName = punyo_link
        self.tare = None

        self.base_area:float = None
        self.base_normal_local:torch.Tensor = torch.tensor(base_normal_local)
    
    def attach(self, model : Union[RigidObjectModel, RobotModelLink]):
        """Attaches the sensor to a punyo model."""
        assert model.getName() == self.attachModelName, "Punyo model name mismatch"
        self.object = model

        mesh = model.geometry().getTriangleMesh()
        self.vertices = mesh.getVertices()
        self.triangles = mesh.getIndices()

        trimesh_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.triangles)        
        # trimesh_mesh.show()
        base_mesh = trimesh_mesh.projected(self.base_normal_local.cpu().numpy())
        # base_mesh.show()
        self.base_area = base_mesh.area

    def measurement_names(self):
        return ['pressure']

    def predict(self, state : SimState) -> torch.Tensor:
        """Called to update the sensor based on the simulation environment."""
        return self.predict_torch(state)
    
    def predict_torch(self, state : SimState) -> torch.Tensor:
        """Called to update the sensor based on the simulation environment.
        This version uses torch operations to predict the observations in a
        differentiable way."""
        assert self.object is not None
        
        tsr_params = {'device': state.device, 'dtype': state.dtype}
        self.base_normal_local = self.base_normal_local.to(**tsr_params)
        
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        pressure = torch.zeros(1) if self.tare is None else torch.from_numpy(self.tare)
        
        pressure = pressure.to(**tsr_params).clone()
        assert pressure.shape == (1,)
        R_local = state.body_transforms[self.attachModelName][:3,:3]
        for body,contact_state in zip(bodies,contact_states):
            # transform contact forces from world to local frame
            forces_local = transform_directions(contact_state.forces,R_local.T)
            pressure -= (1/self.base_area)*torch.sum(forces_local @ self.base_normal_local) 
        return pressure
    
    def measurement_force_jacobian(self, state: SimState) -> Dict[Tuple[str,str],torch.Tensor]:
        """
        Measure the Jacobian of the pressure sensor observation w.r.t. contact forces.
        The result is a map from body pairs to Jacobians. A body pair indicates
        that a body of the sensor is contacting another body and is given
        in the form (sensor_body,other_body).

        A Jacobian must have shape (1,N,3) where N is the number of contact
        points between the two bodies, 3 is the dimension of the force, and
        1 is the dimension of the pressure measurement.
        
        The Jacobian is computed as the derivative of the pressure with respect
        to the contact forces at the contact points. Each entry (i,j,k) is the
        derivative of the pressure with respect to the k'th component of the force
        (in world coordinates) at the j'th contact point.
        The Jacobian is computed as the product of the base normal vector and
        the contact force vector, divided by the base area, for each contact 
        point between the sensor and the other body.

        Args:
            state (SimState): The current simulation state.
        Returns:
            Dict[Tuple[str,str],torch.Tensor]: A dictionary mapping body pairs to Jacobians.        
        """
        assert self.object is not None        
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        R_local = state.body_transforms[self.attachModelName][:3,:3]
        res = {}
        for body,contact_state in zip(bodies,contact_states):
            num_points = contact_state.points.shape[0]
            base_normal_world = R_local @ self.base_normal_local
            # Jacobian shape: (1, num_points, 3)
            jacobian = -(1/self.base_area)*base_normal_world.view(1, 1, 3).repeat(1, num_points, 1)
            res[self.attachModelName,body] = jacobian
        return res
            
    def set_calibration(self, tare):
        self.tare = tare.get('tare',None)
    
    def get_calibration(self):
        if self.tare is None:
            return {}
        else:
            return {'tare': self.tare}