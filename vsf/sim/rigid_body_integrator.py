"""
This module contains a simple explicit Euler rigid body integrator 
that can be used to simulate the motion of a rigid body.
"""

import time
import klampt
import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class RigidBodyIntegrator:
    def __init__(self, mass, inertia_tensor, position, orientation, linear_velocity, angular_velocity):
        """
        Initialize a rigid body with mass and inertia tensor.
        :param mass: Mass of the rigid body (float).
        :param inertia_tensor: 3x3 inertia tensor matrix (numpy array).
        :param position: Initial position (3D vector).
        :param orientation: Initial orientation as a quaternion (w, x, y, z).
        :param linear_velocity: Initial linear velocity (3D vector).
        :param angular_velocity: Initial angular velocity (3D vector).
        """
        self.mass = mass
        self.inertia_tensor = np.array(inertia_tensor, dtype=float)
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)
        self.position = np.array(position, dtype=float)
        self.orientation = Rotation.from_quat(orientation)
        self.linear_velocity = np.array(linear_velocity, dtype=float)
        self.angular_velocity = np.array(angular_velocity, dtype=float)

    def apply_forces(self, contact_points, contact_forces):
        """
        Compute net force and torque from contact points and forces.
        :param contact_points: List of 3D points of contact in world space (Nx3 array).
        :param contact_forces: List of 3D forces at each contact point in world space (Nx3 array).
        :return: Tuple (net_force, net_torque).
        """
        contact_points = np.array(contact_points)
        contact_forces = np.array(contact_forces)
        net_force = np.sum(contact_forces, axis=0)
        net_torque = np.sum(np.cross(contact_points - self.position, contact_forces), axis=0)
        return net_force, net_torque

    def step(self, dt, net_force, net_torque, max_vel = 1.0):
        """
        Perform one time step update using the Euler method.
        
        Change the internal state of the rigid body based on the net force and torque acting on it.

        :param dt: Time step (float).
        :param net_force: Net force acting on the rigid body (3D vector).
        :param net_torque: Net torque acting on the rigid body (3D vector).
        """
        # Linear motion
        acceleration = net_force / self.mass
        self.linear_velocity += acceleration * dt
        self.position += self.linear_velocity * dt

        # Angular motion
        angular_acceleration = self.inertia_tensor_inv @ net_torque
        self.angular_velocity += angular_acceleration * dt
        angular_velocity_norm = np.linalg.norm(self.angular_velocity)
        if angular_velocity_norm > 1e-6:
            delta_rotation = Rotation.from_rotvec(self.angular_velocity * dt)
            self.orientation = delta_rotation * self.orientation
        
        # Limit linear and angular velocities
        linear_velocity_norm = np.linalg.norm(self.linear_velocity)
        if linear_velocity_norm > max_vel:
            self.linear_velocity *= max_vel / linear_velocity_norm
        angular_velocity_norm = np.linalg.norm(self.angular_velocity)
        if angular_velocity_norm > max_vel:
            self.angular_velocity *= max_vel / angular_velocity_norm

    def get_pose(self):
        """
        Get the current pose of the rigid body.
        :return: Tuple of position and orientation (quaternion).
        """
        # TODO: Check if the orientation is correct
        return self.position, self.orientation.as_matrix()