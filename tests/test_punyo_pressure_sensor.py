import pytest
import torch
import numpy as np

from dataclasses import dataclass
from klampt import WorldModel
from klampt.model.geometry import TriangleMesh

# Import the sensor and base classes
from vsf.sensor.punyo_pressure_sensor import PunyoPressureSensor
from vsf.sensor.base_sensor import SimState, ContactState

# TODO: currently every test needs to recreate the sensor object 

#
# TESTS
#
def test_sensor_init():
    """Check initial state of a newly created sensor."""
    sensor = PunyoPressureSensor(name="test_sensor", punyo_link="punyo_link")
    assert sensor.name == "test_sensor"
    assert sensor.attachModelName == "punyo_link"
    assert sensor.tare is None
    
    names = sensor.measurement_names()
    assert names == ['pressure']


def test_attach_success():
    """
    Attaching to the correct punyo object should populate the sensor's geometry attributes.
    """
    sensor = PunyoPressureSensor(name="test_sensor", punyo_link="punyo_link")

    world = WorldModel()
    obj = world.makeRigidObject("punyo_link")
    
    # Create a simple triangular mesh (3 vertices, 1 triangle).
    mesh = TriangleMesh()
    mesh.vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mesh.indices = [[0, 1, 2]]

    # Set the geometry on the rigid object
    obj.geometry().setTriangleMesh(mesh)
    
    sensor.attach(obj)
    assert sensor.object == obj

    
def test_predict():
    """
    Test the sensor's behavior when there are no contacts.
    """
    sensor = PunyoPressureSensor(name="test_sensor", punyo_link="punyo_link", 
                                 base_normal_local=[0.0, 0.0, 1.0])

    world = WorldModel()
    obj = world.makeRigidObject("punyo_link")
    
    # Create a simple triangular mesh (3 vertices, 1 triangle).
    mesh = TriangleMesh()
    mesh.vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mesh.indices = [[0, 1, 2]]

    # Set the geometry on the rigid object
    obj.geometry().setTriangleMesh(mesh)
    
    # Attach the object to the sensor
    sensor.attach(obj)
    assert sensor.base_area is not None, "Base area should be set after attaching the object"
    assert sensor.base_area > 0, "Base area should be positive"
    assert np.allclose(sensor.base_area, 0.5), "Base area should be equal to the area of the triangle"
    assert sensor.base_normal_local is not None, "Base normal should be set after attaching the object"
    assert isinstance(sensor.base_normal_local, torch.Tensor), "Base normal should be a tensor"
    assert sensor.base_normal_local.shape == (3,), "Base normal should be a 3D vector"
    assert torch.allclose(sensor.base_normal_local, torch.tensor([0.0, 0.0, 1.0])), "Base normal should be [0, 0, 1]"

    # Create an empty contact state
    contact_state = ContactState(
        points=torch.zeros((0, 3)),  # No contact points
        forces=torch.zeros((0, 3)),  # No forces
        elems1=torch.zeros((0,), dtype=torch.int64),  # No elements
        elems2=None
    )
    sim_state = SimState()
    sim_state.contacts = {("punyo_link", "other_body"): contact_state}
    sim_state.body_transforms = {"punyo_link": torch.eye(4)}
    sim_state.body_states = {"punyo_link": torch.zeros(6)}  # Dummy state
    
    measurement = sensor.predict(sim_state)
    assert measurement.shape == (1,)
    assert torch.allclose(measurement, torch.tensor([0.0])), "Expected zero pressure when no contact"
    
    contact_state = ContactState(
        points=torch.zeros((1, 3)),  # One contact point
        forces=-torch.ones((1, 3)),  # Zero force
        elems1=torch.tensor([0]),  # One element
        elems2=None
        
    )
    sim_state = SimState()
    sim_state.contacts = {("punyo_link", "other_body"): contact_state}
    sim_state.body_transforms = {"punyo_link": torch.eye(4)}
    sim_state.body_states = {"punyo_link": torch.zeros(6)}  # Dummy state
    
    measurement = sensor.predict(sim_state)
    assert measurement.shape == (1,)
    print(measurement)
    assert torch.allclose(measurement, (1/0.5)*torch.tensor([1.0])), "Expected zero pressure when no contact"
    
    jacobian_dict = sensor.measurement_force_jacobian(sim_state)
    jacobian_key = ('punyo_link', 'other_body')
    assert jacobian_key in jacobian_dict
    assert jacobian_dict[jacobian_key].shape == (1, 1, 3), f"Unexpected shape {jacobian_dict[jacobian_key].shape} for Jacobian"
    assert len(jacobian_dict) == 1
    jacobian_gt = torch.zeros((1, 1, 3))
    jacobian_gt[0, 0, :] = -(1/0.5)*torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(jacobian_dict[jacobian_key], jacobian_gt), "Jacobian does not match expected value"


  
def test_predict_random():
    """
    Test the sensor's behavior when there are no contacts.
    """
    sensor = PunyoPressureSensor(name="test_sensor", punyo_link="punyo_link", 
                                 base_normal_local=[0.0, 0.0, 1.0])

    world = WorldModel()
    obj = world.makeRigidObject("punyo_link")
    
    # Create a simple triangular mesh (3 vertices, 1 triangle).
    mesh = TriangleMesh()
    # Create a simple triangular mesh (3 vertices, 1 triangle).
    mesh = TriangleMesh()
    mesh.vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    mesh.indices = [[0, 1, 2]]

    # Set the geometry on the rigid object
    obj.geometry().setTriangleMesh(mesh)

    # Attach the object to the sensor
    sensor.attach(obj)
    
    for num_contacts in range(1, 100, 10):        
        
        from scipy.spatial.transform import Rotation
        R = Rotation.random().as_matrix()  # Returns a 3x3 rotation matrix
        t = torch.rand((3,))  # Random translation vector
        
        H = torch.eye(4)
        H[:3, :3] = torch.tensor(R)
        H[:3, 3] = t
        
        forces = torch.rand((num_contacts, 3))  # Random forces

        contact_state = ContactState(
            points=torch.rand((num_contacts, 3)),  # One contact point
            forces=forces,  # Zero force
            elems1=torch.zeros((num_contacts,)),  # One element
            elems2=None
        )

        sim_state = SimState()
        sim_state.contacts = { ("punyo_link", "other_body"): contact_state }
        sim_state.body_transforms = { "punyo_link": H }
        sim_state.body_states = { "punyo_link": torch.zeros(6) }  # Dummy state
    
        measurement = sensor.predict(sim_state)
        assert measurement.shape == (1,)

        jacobian_dict = sensor.measurement_force_jacobian(sim_state)
        jacobian_key = ('punyo_link', 'other_body')
        jacobian = jacobian_dict[jacobian_key]

        measurement_hat = torch.einsum('opc,pc->o', jacobian, forces)
        assert torch.allclose(measurement_hat, measurement), \
            f"Jacobian does not match expected value: expected {measurement_hat}, got {measurement}"
