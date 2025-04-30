# Base class for sensor model
import torch
import numpy as np
from klampt import RobotModel
from .base_sensor import BaseSensor,SimState
from typing import Dict,Tuple

class JointTorqueSensor(BaseSensor):
    """Joint torque sensor model.
    
    We assume the sensors are attached to the joint axes of certain links.
    These links are marked by link_names, list of link indices in the robot model.

    If you would like to collide with a different set of links, these can
    be provided in collision_links.

    Attributes:
        name (str): The name of the sensor.
        attachModelName (str): The name of the robot the sensor is attached to.
        link_names (list[str]): The names or indices of links for which torques should
            be reported.
        collision_links (list[str]): The names or indices of links that should be used
            for collision detection.
        gravity (tuple): The gravity vector.
        robot (RobotModel): The robot model the sensor is attached to.
        tare (np.ndarray): The tare value for the sensor.
        gravity_pred_at_tare (np.ndarray): The predicted gravity value at the time of
            tare.
        noise_stddev (float): The standard deviation of the joint torque noise.
            Default value is 1.0, assuming unit is Nm.
    """
    
    def __init__(self, name:str, robot_name: str, link_names: list[str], 
                 gravity = (0,0,-9.8), noise_stddev = 1.0):
        self.name: str = name
        self.attachModelName = robot_name
        self.link_names = link_names
        self.collision_links = link_names[-4:]  # HACK: last 4 links are collision links
        self.gravity = gravity
        self.robot = None
        self.tare = None
        self.gravity_pred_at_tare = None
        self.noise_stddev = noise_stddev
    
    def attach(self, obj : RobotModel):
        """Attaches the sensor to a robot model."""
        assert isinstance(obj, RobotModel), "JointTorqueSensor can only be attached to a RobotModel"
        assert obj.getName() == self.attachModelName, "Robot name mismatch"
        self.robot = obj
    
    def measurement_names(self):
        return [n+'_torque' for n in self.link_names]

    def measurement_errors(self):
        return np.array([self.noise_stddev]*len(self.link_names))

    def predict(self, state : SimState) -> torch.Tensor:
        assert self.robot is not None, "Sensor is not attached to a robot model"
        G = self.robot.getGravityForces(self.gravity)  #assumes robot config is updated
        joint_torques = -np.array([G[self.robot.link(n).index] for n in self.link_names])
        if self.tare is not None:
            # subtract the tare value
            if self.gravity_pred_at_tare is None:
                #the tare value includes gravity, so we need to subtract the gravity
                self.gravity_pred_at_tare = joint_torques
            joint_torques += self.tare - self.gravity_pred_at_tare
        joint_torques = torch.from_numpy(joint_torques).to(state.device)

        J = len(self.link_names)
        contact_jacobians = self.measurement_force_jacobian(state, verbose=False)
        
        for (link_name,body_name), contact_jacobian in contact_jacobians.items():
            if contact_jacobian.shape[1] == 0:
                continue
            contact_state = state.contacts[link_name,body_name]
            points = contact_state.points
            forces = contact_state.forces
            assert contact_jacobian.shape[0] == len(self.link_names)
            assert len(points) == len(forces)
            assert len(points) == contact_jacobian.shape[1]
            joint_torques += (contact_jacobian.reshape(J, -1) @ forces.reshape(-1))
        return joint_torques
    
    def predict_torch(self, state : SimState) -> torch.Tensor:
        """Same as predict but preserves torch operations"""
        assert self.robot is not None, "Sensor is not attached to a robot model"
        device = state.body_transforms[self.link_names[0]].device
        G = self.robot.getGravityForces(self.gravity)  #assumes robot config is updated
        joint_torques = -torch.tensor([G[self.robot.link(n).index] for n in self.link_names],device=device)
        if self.tare is not None:
            tare_tensor = torch.from_numpy(self.tare).to(device)
            # subtract the tare value
            if self.gravity_pred_at_tare is None:
                #the tare value includes gravity, so we need to subtract the gravity
                self.gravity_pred_at_tare = joint_torques
            gravity_pred_tensor = torch.from_numpy(self.gravity_pred_at_tare).to(device)
            joint_torques += tare_tensor - gravity_pred_tensor
        for link_name_i in self.collision_links:
            i = self.robot.link(link_name_i).index
            contact_points, contact_forces = state.contacts_on_body(link_name_i, local=False)

            for j, link_name_j in enumerate(self.link_names):
                if j > i:  #no jacobian entry for this case,
                    break 
                link_j = self.robot.link(link_name_j)
                link_axis = link_j.getAxis()
                world_link_origin = torch.tensor(link_j.getWorldPosition([0, 0, 0]), device=device).unsqueeze(0)
                world_link_axis = torch.tensor(link_j.getWorldDirection(link_axis), device=device).unsqueeze(0)

                joint_torques[j] += torch.sum(torch.cross(world_link_axis, contact_forces) * (world_link_origin - contact_points))
        return joint_torques

    def measurement_force_jacobian(self, state: SimState, verbose=False) -> Dict[Tuple[str,str],torch.Tensor]:
        """Compute the Jacobian matrix d measurement / d contact force
        from the current state.

        Each result has shape num_joints x N x 3, where N is the number of contact
        points and num_joints is the number of joints in the joint torque sensor.
        """
        if len(state.contacts) == 0:
            return {}
        num_joints = len(self.link_names)

        res = {}
        for i,link_name in enumerate(self.collision_links):
            bodies = state.bodies_in_contact(link_name)
            contact_states = state.contact_states_on_body(link_name)
            if len(bodies) == 0:
                continue
            for body_name,contact_state in zip(bodies,contact_states):
                if verbose:
                    print(len(contact_state.points),"contacts on link index:", link_name,"vs",body_name)

                link_i = self.robot.link(link_name)
                link_Jac = np.zeros((num_joints, len(contact_state.points), 3))
                for query_idx in range(len(self.link_names)):
                    link_j = self.robot.link(self.link_names[query_idx])
                    if link_j.getIndex() > link_i.getIndex():  #assume serial robot
                        continue
                    origin_world = link_j.getTransform()[1]
                    axis_world = link_j.getWorldDirection(link_j.getAxis())
                    # world dr shape: N x 3
                    world_dr = contact_state.points - np.array(origin_world)
                    # axis x dr shape: N x 3
                    axis_x_dr = np.cross(np.array(axis_world), world_dr)
                    link_Jac[query_idx,:,:] = axis_x_dr

                res[(link_name, body_name)] = torch.from_numpy(link_Jac)
        return res

    def set_calibration(self, calib_dict):
        self.tare = calib_dict.get('tare',None)
        self.gravity_pred_at_tare = calib_dict.get('tare_sim_avg',None)

    def get_calibration(self):
        return {'tare':self.tare, 'gravity_pred_at_tare':self.gravity_pred_at_tare}