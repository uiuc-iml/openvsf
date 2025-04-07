import pytest
import numpy as np
import torch

from klampt import WorldModel
from klampt.model.geometry import TriangleMesh

# Adjust these imports to reflect your actual code structure
from vsf.sensor.punyo_dense_force_sensor import (
    PunyoDenseForceSensor,
    barycentric_weights
)
from vsf.sensor.base_sensor import SimState
from vsf.utils.data_utils import transform_points, transform_directions


# ------------------------------------------------------------------------------
#                      Fixtures: Klampt World with a single RigidObject
# ------------------------------------------------------------------------------
@pytest.fixture
def world_with_single_rigid_object():
    """
    Creates a Klampt WorldModel, adds a RigidObjectModel with one triangle:
      vertices: [(0,0,0),(1,0,0),(0,1,0)]
      indices:  [(0,1,2)]

    Returns the entire WorldModel object. Tests can then do:
      rigid_obj = world_with_single_rigid_object.getRigidObjects()[0]
    """
    world = WorldModel()

    # Create an empty rigid object in the world
    rigid_obj = world.makeRigidObject("test_rigid_obj")

    # Set up a single triangle on the object's geometry
    mesh = TriangleMesh()
    mesh.setVertices([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    mesh.setIndices([
        [0, 1, 2]
    ])
    rigid_obj.geometry().setTriangleMesh(mesh)

    return world


# ------------------------------------------------------------------------------
#                           Minimal Mock SimState
# ------------------------------------------------------------------------------
def mock_sim_state(
    device: torch.device,
    attach_model_name: str,
    contact_forces,
    elems1,
    elems2 = None,
    rotation: np.ndarray = None,
    contact_points: np.ndarray = None
):
    """
    Creates a minimal SimState with:
        - .device
        - .body_transforms: {attach_model_name: 4x4 transform}
        - .bodies_in_contact(attach_model_name)
        - .contact_states_on_body(attach_model_name)

    contact_forces: Nx3 forces
    elems1: list of vertex indices in contact
    elems2: list of VSF points indices in contact
    rotation: optional 3x3 rotation for body_transforms
    contact_points: optional Nx3 points in world coordinates
    """

    class MockContactState:
        def __init__(self, forces, elems1, elems2=None, points=None):
            self.elems1 = elems1
            self.elems2 = elems2
            
            if elems1 is None:
                self.elems1 = []
            if elems2 is None:
                self.elems2 = []
            
            num_contacts = max(len(self.elems1), len(self.elems2))            
            if points is None:
                self.points = torch.zeros((num_contacts, 3), dtype=torch.float32, device=device)
            else:
                self.points = torch.tensor(points, dtype=torch.float32, device=device)

            self.forces = torch.tensor(forces, dtype=torch.float32, device=device)

    class MockSimState(SimState):
        def __init__(self):
            # Build a 4x4 transform with optional rotation
            mat = torch.eye(4, dtype=torch.float32, device=device)
            if rotation is not None:
                mat[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
            self.body_transforms = {attach_model_name: mat}

            self._bodies = ["some_other_body"]  # placeholder name
            # Single "contact" with the specified elems/forces
            self._contact_states = [
                MockContactState(contact_forces, elems1, elems2, contact_points)
            ]

        def bodies_in_contact(self, body_name):
            if body_name == attach_model_name:
                return self._bodies
            return []

        def contact_states_on_body(self, body_name):
            if body_name == attach_model_name:
                return self._contact_states
            return []

    return MockSimState()


# ------------------------------------------------------------------------------
#                            Tests for barycentric_weights
# ------------------------------------------------------------------------------
def test_barycentric_weights_on_triangle(world_with_single_rigid_object: WorldModel):
    """
    Test barycentric_weights using the actual mesh from the single RigidObject in the fixture.
    We'll evaluate a few points (vertex corners, an interior point) 
    and check the results.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]

    # Extract geometry from RigidObjectModel
    mesh_data = rigid_object.geometry().getTriangleMesh()
    vertices = np.array(mesh_data.getVertices())
    indices = np.array(mesh_data.getIndices())

    # Points: exactly the 3 vertices
    points = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0 => barycentric ~ [1, 0, 0]
        [1.0, 0.0, 0.0],  # Vertex 1 => barycentric ~ [0, 1, 0]
        [0.0, 1.0, 0.0],  # Vertex 2 => barycentric ~ [0, 0, 1]
    ])
    weights = barycentric_weights(vertices, indices, points)
    assert weights.shape == (3, 3), "Should have shape (num_points, num_vertices)."

    np.testing.assert_allclose(weights[0], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(weights[1], [0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(weights[2], [0, 0, 1], atol=1e-7)

    # Interior point (0.3, 0.3, 0)
    points_interior = np.array([
        [0.3, 0.3, 0.0]
    ])
    weights_interior = barycentric_weights(vertices, indices, points_interior)
    assert weights_interior.shape == (1, 3)
    w = weights_interior[0]
    # Barycentric coords sum to 1
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
    # They should each be within [0,1]
    assert (w >= 0).all() and (w <= 1).all()
    
    # Middle of the triangle (0.5, 0.5, 0)
    points_middle = np.array([
        [0.5, 0.5, 0.0]
    ])
    weights_middle = barycentric_weights(vertices, indices, points_middle)
    assert weights_middle.shape == (1, 3)
    np.testing.assert_allclose(weights_middle[0], [0.0, 0.5, 0.5], atol=1e-7)


# ------------------------------------------------------------------------------
#               Tests for PunyoDenseForceSensor with a Single Triangle
# ------------------------------------------------------------------------------
def test_attach_sensor(world_with_single_rigid_object: WorldModel):
    """
    Check that attach(...) properly reads vertices and indices 
    from the RigidObject's geometry.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")

    sensor.attach(rigid_object)

    # Compare sensor's stored vertices/indices to the object's geometry
    mesh_data = rigid_object.geometry().getTriangleMesh()
    np.testing.assert_allclose(sensor.vertices, mesh_data.getVertices())
    np.testing.assert_array_equal(sensor.triangles, mesh_data.getIndices())


def test_measurement_names(world_with_single_rigid_object: WorldModel):
    """
    measurement_names() should list 3 * num_vertices strings.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")
    sensor.attach(rigid_object)

    names = sensor.measurement_names()
    n_verts = len(sensor.vertices)
    assert len(names) == 3 * n_verts, "Should have 3 names per vertex."


def test_predict_identity_transform(world_with_single_rigid_object: WorldModel):
    """
    If the transform is identity, the predicted force = sum of contact forces 
    at the specified vertices, no rotation is applied.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")
    sensor.attach(rigid_object)

    # Suppose contact on vertex #1 with force [0, 0, 2].
    elems1 = [1]
    contact_forces = [[0.0, 0.0, 2.0]]
    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        elems1=elems1,
        contact_forces=contact_forces,
        rotation=None  # identity
    )

    force_tensor = sensor.predict_torch(state)
    n_verts = len(sensor.vertices)
    assert force_tensor.shape == (3 * n_verts,)

    force_np = force_tensor.cpu().numpy().reshape(n_verts, 3)
    # We expect [0,0,2] at vertex #1, zeros elsewhere
    np.testing.assert_allclose(force_np[0], [0,0,0], atol=1e-6)
    np.testing.assert_allclose(force_np[1], [0,0,2], atol=1e-6)
    np.testing.assert_allclose(force_np[2], [0,0,0], atol=1e-6)


def test_predict_with_rotation(world_with_single_rigid_object: WorldModel):
    """
    If there's a rotation R, the code transforms contact forces by R^T. 
    We'll test a rotation about Z by +90 deg, so R^T is effectively -90 deg.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")
    sensor.attach(rigid_object)

    # Rotation about Z by +90 deg
    # R = [[ 0, -1, 0],
    #      [ 1,  0, 0],
    #      [ 0,  0, 1]]
    R = np.array([
        [ 0, -1,  0],
        [ 1,  0,  0],
        [ 0,  0,  1]
    ], dtype=float)

    # Contact force [1,0,0]
    elems1 = [0]
    contact_forces = [[1.0, 0.0, 0.0]]

    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        elems1=elems1,
        contact_forces=contact_forces,
        rotation=R
    )
    force_tensor = sensor.predict_torch(state)
    n_verts = len(sensor.vertices)
    force_np = force_tensor.cpu().numpy().reshape(n_verts, 3)
    
    # (1,0,0) under R^T => (0,-1,0)
    np.testing.assert_allclose(force_np[0, :], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(force_np[1, :], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(force_np[2, :], [0, 0, 0], atol=1e-6)


def test_predict_surface_points(world_with_single_rigid_object: WorldModel):
    """
    If there's a rotation R, the code transforms contact forces by R^T. 
    We'll test a rotation about Z by +90 deg, so R^T is effectively -90 deg.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")
    sensor.attach(rigid_object)

    # Rotation about Z by +90 deg
    # R = [[ 0, -1, 0],
    #      [ 1,  0, 0],
    #      [ 0,  0, 1]]
    R = np.array([
        [ 1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0,  1]
    ], dtype=float)

    # Contact force [1,0,0]
    elems1 = None
    elems2 = [0]
    contact_forces = [[1.0, 0.0, 0.0]]
    contact_points = [[0.5, 0.5, 0.0]]  # Point in the middle of the triangle

    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        contact_forces=contact_forces,
        elems1=elems1,
        elems2=elems2,
        rotation=R, 
        contact_points=contact_points
    )
    force_tensor = sensor.predict_torch(state)
    n_verts = len(sensor.vertices)
    force_np = force_tensor.cpu().numpy().reshape(n_verts, 3)

    # (1,0,0) distributed to vertices according to weight [0,0.5,0.5]
    np.testing.assert_allclose(force_np[0, :], [0, 0.0, 0], atol=1e-6)
    np.testing.assert_allclose(force_np[1, :], [0.5, 0.0, 0], atol=1e-6)
    np.testing.assert_allclose(force_np[2, :], [0.5, 0.0, 0], atol=1e-6)


def test_measurement_force_jacobian(world_with_single_rigid_object: WorldModel):
    """
    Checks jacobian shape for a single contact at a single vertex.
    """
    rigid_object = world_with_single_rigid_object.getRigidObjects()[0]
    sensor = PunyoDenseForceSensor("punyo_sensor", "link0")
    sensor.attach(rigid_object)

    # Single contact on vertex 0 => for test
    elems1 = [0]
    contact_forces = [[0,0,1]]
    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        elems1=elems1,
        contact_forces=contact_forces
    )
    jac = sensor.measurement_force_jacobian(state)
    # We expect the key: ("link0", "some_other_body")
    assert ("link0", "some_other_body") in jac

    J = jac[("link0", "some_other_body")]
    # Code doc suggests shape: (num_vertices*3, num_points, 3).
    num_vertices = len(sensor.vertices)
    num_contacts = len(elems1)  # 1
    expected_shape = (3 * num_vertices, num_contacts, 3)
    assert J.shape == expected_shape, f"Jacobian shape mismatch: got {J.shape} vs {expected_shape}"

    # For identity rotation, we expect an identity block in the slice for vertex 0
    block_0 = J[0:3, 0, :]
    np.testing.assert_allclose(block_0.cpu().numpy(), np.eye(3), atol=1e-6)

    # The rest should be zero
    block_rest = J[3:, 0, :]
    np.testing.assert_allclose(block_rest.cpu().numpy(), 0.0, atol=1e-6)
    
    # Contact on non-vertex point
    elems1 = []
    elems2 = [0]
    contact_forces = [[0,0,1]]
    contact_points = [[0.5, 0.5, 0.0]]  # Point in the middle of the triangle
    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        contact_forces=contact_forces,
        elems1=elems1,
        elems2=elems2,
        contact_points=contact_points
    )
    jac = sensor.measurement_force_jacobian(state)
    # We expect the key: ("link0", "some_other_body")
    assert ("link0", "some_other_body") in jac

    J = jac[("link0", "some_other_body")]
    # Code doc suggests shape: (num_vertices*3, num_points, 3).
    num_vertices = len(sensor.vertices)
    num_contacts = len(elems2)  # 1
    expected_shape = (3 * num_vertices, num_contacts, 3)
    assert J.shape == expected_shape, f"Jacobian shape mismatch: got {J.shape} vs {expected_shape}"

    block_0 = J[0:3, 0, :]
    np.testing.assert_allclose(block_0.cpu().numpy(), 0.0, atol=1e-6)

    block_1 = J[3:6, 0, :]
    np.testing.assert_allclose(block_1.cpu().numpy(), 0.5*np.eye(3), atol=1e-6)
    
    block_2 = J[6:9, 0, :]
    np.testing.assert_allclose(block_2.cpu().numpy(), 0.5*np.eye(3), atol=1e-6)
    
    # Test rotated case
    elems1 = []
    elems2 = [0]
    contact_forces = [[0,0,1]]
    contact_points = [[-0.5, 0.5, 0.0]]  # Point in the middle of the triangle
    rotation = np.array([
        [ 0, -1,  0],
        [ 1,  0,  0],
        [ 0,  0,  1]
    ], dtype=float)
    state = mock_sim_state(
        device=torch.device("cpu"),
        attach_model_name="link0",
        contact_forces=contact_forces,
        elems1=elems1,
        elems2=elems2,
        contact_points=contact_points, 
        rotation=rotation
    )
    jac = sensor.measurement_force_jacobian(state)
    # We expect the key: ("link0", "some_other_body")
    assert ("link0", "some_other_body") in jac

    J = jac[("link0", "some_other_body")]
    # Code doc suggests shape: (num_vertices*3, num_points, 3).
    num_vertices = len(sensor.vertices)
    num_contacts = len(elems2)  # 1
    expected_shape = (3 * num_vertices, num_contacts, 3)
    assert J.shape == expected_shape, f"Jacobian shape mismatch: got {J.shape} vs {expected_shape}"

    print('J:', J)

    Rlocal = rotation.T

    block_0 = J[0:3, 0, :]
    np.testing.assert_allclose(block_0.cpu().numpy(), 0.0, atol=1e-6)

    block_1 = J[3:6, 0, :]
    np.testing.assert_allclose(block_1.cpu().numpy(), 0.5*Rlocal, atol=1e-6)
    
    block_2 = J[6:9, 0, :]
    np.testing.assert_allclose(block_2.cpu().numpy(), 0.5*Rlocal, atol=1e-6)
