from __future__ import annotations
import torch
from klampt import RobotModel, RigidObjectModel, RobotModelLink
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, field

@dataclass
class ContactState:
    """A representation of the forces and contact points between two
    bodies.
    """
    points: torch.Tensor  # Nx3 array of contact points, in world coordinates
    forces: torch.Tensor  # Nx3 array of contact forces, in world coordinates applied to object 1
    elems1: Optional[torch.LongTensor]=None  # length N array of contact elements of object 1
    elems2: Optional[torch.LongTensor]=None  # length N array of contact elements of object 2

    def to(self, device) -> ContactState:
        return ContactState(self.points.to(device),self.forces.to(device),(None if self.elems1 is None else self.elems1.to(device)),(None if self.elems2 is None else self.elems2.to(device)))


@dataclass
class SimState:
    """A representation of the simulator state used for sensor simulation.
    
    Forces are not assumed to be symmetric, so if you wish to store contact
    forces applied to a with equal and opposite forces on b, you must store
    both (a,b) and (b,a) in the contact_forces dictionary. 
    """
    body_transforms: Dict[str, torch.Tensor] = field(default_factory=dict)  
    """Named rigid body homogeneous transforms"""
    body_states: Dict[str, torch.Tensor] = field(default_factory=dict)      
    """Named body states (e.g., robot configs, deformable states)"""
    contacts : Dict[Tuple[str,str], ContactState] = field(default_factory=dict)  
    """Contact state between pairs of rigid bodies"""

    def subgraph(self, names : List[str]) -> SimState:
        """Returns a subgraph of the simulation state."""
        names = set(names)
        return SimState({k:t for (k,t) in self.body_transforms.items() if k in names}, {k:x for (k,x) in self.body_states.items()}, {(k1,k2):c for ((k1,k2),c) in self.contacts if k1 in names and k2 in names})

    def bodies_in_contact(self, name : str, reverse = False) -> List[str]:
        """Returns the list of bodies that `name` is in contact with
        (or that are in contact with `name` if `reverse=True`).
        """
        if name not in self.body_transforms:
            raise ValueError(f"Body {name} not found in state")
        if reverse:
            return [a for (a,b) in self.contacts.keys() if b == name]
        else:
            return [b for (a,b) in self.contacts.keys() if a == name]

    def contact_states_on_body(self, name : str, reverse = False) -> List[ContactState]:
        """Returns a list of contact states affecting a given body.
        
        These will match the order given by `bodies_in_contact(name,reverse)`.
        
        If reverse = False, all the forces are affecting the body.
        Otherwise, all the forces are affected by the body.
        """
        if name not in self.body_transforms:
            raise ValueError(f"Body {name} not found in state")
        if reverse:
            return [c for ((a,b),c) in self.contacts.items() if b == name]
        else:
            return [c for ((a,b),c) in self.contacts.items() if a == name]

    def contacts_on_body(self, name : str, local : bool = True) -> Tuple[torch.Tensor,torch.Tensor]:
        """Returns a concatenated list of all contacts affecting a given body.
        
        Args:
            name (str): the name of the body
            local (bool, optional): if True, returns the contact points and
                forces in local coordinates. Defaults to True.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: the contact points and forces
        """
        if name not in self.body_transforms:
            raise ValueError(f"Body {name} not found in state")
        pts = []
        forces = []
        for (a,b),cps in self.contacts.items():
            if a == name:
                pts.append(cps.points)
                forces.append(cps.forces)
        pts = torch.concat(pts) if pts else torch.empty(0,3,device=self.device)
        forces = torch.concat(forces) if forces else torch.empty(0,3,device=next(iter(self.body_transforms.values())).device)
        if local:
            T = self.body_transforms[name].to(pts.device).to(pts.dtype)
            Rinv = T[:3,:3].T
            tinv = -torch.matmul(Rinv,T[:3,3])
            pts = torch.matmul(pts,Rinv.T) + tinv
            forces = torch.matmul(forces,Rinv.T)
        return pts,forces

    @property
    def device(self):
        if len(self.body_transforms) == 0:
            return torch.device('cpu')
        return next(iter(self.body_transforms.values())).device
    
    @property
    def dtype(self):
        if len(self.body_transforms) == 0:
            return torch.float32
        return next(iter(self.body_transforms.values())).dtype


class BaseSensor:
    """A sensor model that predicts a real sensor's observations given a
    simulation state. 

    This base class should be extended to implement your sensor model.
    At minimum, the `measurement_names` method, and either the `predict_torch`
    method or both the `predict` and `measurement_force_jacobian` methods must
    be overridden.

    A sensor calibration method can be used to configure how the sensor
    operates.  For example, such a method can tare the sensor, set the
    measurement range, or set the sensor's noise model.  A `BaseCalibrator`
    object with a matching interface should be used to configure the sensor
    based on the calibration data.

    Attributes:
        name (str): The name of the sensor.
        attachModelName (str): The name of the body or robot the sensor is
        attached to.
        
    """
    
    def __init__(self):
        self.name: str = ""
        self.attachModelName = ""

    def attach(self, model: Union[RigidObjectModel,RobotModelLink,RobotModel]):
        """Attaches the sensor to a model."""
        pass

    def measurement_names(self) -> List[str]:
        """Returns a list of names for each measurement the sensor provides."""
        return []
    
    def measurement_errors(self) -> List[float]:
        """Returns a list of standard deviations for each measurement
        the sensor provides.  If None is returned, then the sensor is assumed
        to have isotropic noise."""
        return None

    def predict(self, state : SimState) -> torch.Tensor:
        """Returns the sensor's estimated observation based on the simulation
        environment.  The current attached model is assumed to be updated to
        the correct configuration."""
        raise NotImplementedError

    def predict_torch(self, state : SimState) -> torch.Tensor:
        """Returns the sensor's estimated observation based on the simulation
        environment. The current attached model is assumed to be updated to
        the correct configuration.
        
        Should use torch operations to predict the observations in a
        differentiable way.
        """
        raise NotImplementedError

    def measurement_force_jacobian(self, state : SimState) -> Dict[Tuple[str,str],torch.Tensor]:
        """Called to compute the Jacobian of the tactile sensor observation
        w.r.t. contact forces.
        
        The result is a map from body pairs to Jacobians.  A body pair indicates
        that a body of the sensor is contacting another body and is given
        in the form (sensor_body,other_body).
        
        A Jacobian must have shape (#meas, N, 3) where N is the number of contact
        points between the two bodies, 3 is the dimension of the force, and #meas
        is the number of measurements.  Each entry (i,j,k) is the derivative of the
        i'th measurement with respect to the k'th component of the force (in world
        coordinates) at the j'th contact point.
        
        NOTE: this function assumes the world state has been configured 
        consistently with the SimState, since SimState does not save the information 
        of robot/object state.
        """
        pass

    def update(self, state : SimState):
        """Called to update the sensor based on the simulation environment."""
        pass
    
    def contact_bodies(self, state : SimState) -> List[Tuple[str,str]]:
        """Returns a list of bodies that correspond in order with the points
        predicted via measurement_force_jacobian predictions."""
        pass

    def set_calibration(self, calib_dict : dict):
        """Sets the sensor calibration.  This will be used by a corresponding
        BaseCalibrator object to configure the sensor."""
        pass

    def get_calibration(self) -> dict:
        """Returns the sensor calibration."""
        return {}

    def reset(self) -> None:
        """Resets the sensor to an initial state if it is stateful."""
        pass

    def get_internal_state(self) -> dict:
        """Returns any additional internal state of the sensor."""
        return {}

    def set_internal_state(self, state: dict) -> None:
        """Sets the sensor's internal state."""
        pass


