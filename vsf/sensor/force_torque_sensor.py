# Base class for sensor model
import torch
import numpy as np
from klampt import RobotModelLink
from .base_sensor import BaseSensor, SimState
from ..utils.data_utils import transform_directions
from typing import Dict,Tuple


class ForceTorqueSensor(BaseSensor):
    """Force torque sensor model. The force-torque sensor is attached to a robot link or rigid object.

    Attributes:
        name (str): The name of the sensor.
        attachModelName (str): The name of the robot link that the sensor is attached to.
    """
    
    def __init__(self, name:str, model_name: str):
        self.name: str = name
        self.attachModelName = model_name
        self.link = None
        self.tare = None

    def attach(self, object):
        assert isinstance(object,RobotModelLink)
        self.link = object

    def measurement_names(self):
        return ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    def predict(self, state: SimState) -> torch.Tensor:
        return self.predict_torch(state)

    def predict_torch(self, state: SimState) -> torch.Tensor:
        assert self.link is not None
        contact_points, contact_forces = state.contacts_on_body(self.attachModelName,False)

        Rlocal = state.body_transforms[self.attachModelName][:3,:3].T
        origin = state.body_transforms[self.attachModelName][:3,3]
        force = Rlocal @ torch.sum(contact_forces, dim=0)
        torque = Rlocal @ torch.sum(torch.cross(contact_points - origin, contact_forces, dim=1),dim=0)
        ft = torch.concatenate([force, torque])
        if self.tare is not None:
            ft += self.tare
        return ft
    
    def measurement_force_jacobian(self, state: SimState) -> Dict[Tuple[str,str],torch.Tensor]:
        """Compute the contact Jacobian matrix from current state.
        Result has shape 6 x N x 3, where N is the number of contact points."""
        assert self.link is not None
        bodies = state.bodies_in_contact(self.attachModelName)
        contact_states = state.contact_states_on_body(self.attachModelName)
        Rlocal = state.body_transforms[self.attachModelName][:3,:3].T
        origin = state.body_transforms[self.attachModelName][:3,3]
        res = {}
        for (body,contact_state) in zip(bodies,contact_states):
            num_pts = contact_state.points.shape[0]
            
            world_dr = contact_state.points - origin
            local_dr = world_dr @ Rlocal.T
            
            force_jacobian = torch.tile(Rlocal.reshape(3,1,3), (1, num_pts, 1))
            # create skew symmetric matrix from cross product vector
            torque_jacobian = torch.zeros((3, num_pts, 3),device=state.device)
            torque_jacobian[1, :, 0] = -local_dr[:, 2]
            torque_jacobian[2, :, 0] = local_dr[:, 1]
            torque_jacobian[0, :, 1] = local_dr[:, 2]
            torque_jacobian[2, :, 1] = -local_dr[:, 0]
            torque_jacobian[0, :, 2] = -local_dr[:, 1]
            torque_jacobian[1, :, 2] = local_dr[:, 0]
            res[self.attachModelName,body] = torch.concatenate([force_jacobian, torque_jacobian], dim=2)
        return res

    def set_calibration(self, calib_dict : dict):
        self.tare = calib_dict.get('tare',None)
        
    def get_calibration(self):
        res = {}
        if self.tare is not None: res['tare'] = self.tare
        return res