import time
import numpy as np
import torch
import pytest

# Import the estimator and the observation model.
from vsf.estimator.recursive_optimizer import DiagonalEKF, diag_AtB
from vsf.estimator.recursive_optimizer import ObservationLinearization

def create_obs(n: int, update_format: str = 'information') -> ObservationLinearization:
    """
    Create an observation that observes all n state elements.
    Uses an identity observation matrix, zero bias, and unit variance.
    """
    state_indices = torch.arange(n, dtype=torch.long)
    matrix = torch.eye(n, dtype=torch.double)
    bias = torch.zeros(n, dtype=torch.double)
    # Use a 1D variance vector (diagonal form)
    var = torch.ones(n, dtype=torch.double)
    return ObservationLinearization(matrix=matrix, var=var, bias=bias, state_indices=state_indices)

# --- Fixtures ---

@pytest.fixture
def base_estimator_info():
    """
    Create a basic DiagonalEKF estimator with a 3-element heterogeneous state only.
    For simplicity we do not include a latent part (latent_basis=None).
    """
    max_dim = 3
    # Use nonzero initial mean and variance for heterogeneous state.
    x_mu = torch.tensor([0.5, 0.2, 0.8], dtype=torch.double)
    x_var = torch.tensor([1.0, 1.0, 1.0], dtype=torch.double)
    # No latent component.
    estimator = DiagonalEKF(max_dim,
                            x_mu=x_mu, x_var=x_var,
                            latent_mu=0.0, latent_var=0.0,
                            latent_basis=None,
                            max_buffer_len=5,
                            num_replay_update=3,
                            non_negative=True,
                            update_format='information')
    # Override the diag_AtB function if the estimator does not have its own version.
    estimator.diag_AtB = diag_AtB
    return estimator

# --- Test Cases ---

def test_diag_ekf_step_information(base_estimator_info):
    """Test a single update step using the information filter update format."""
    estimator = base_estimator_info
    estimator.update_format = 'information'
    n = 3
    obs_model = create_obs(n, update_format='information')
    # Create a measurement that yields a small positive residual.
    measurement = estimator.y_mu[obs_model.state_indices] + 0.1

    # Save a copy of the variance to check that replay steps do not modify variance.
    y_var_before = estimator.y_var.clone()
    # Run an update step (non-replay).
    estimator.diag_ekf_step(obs_model, measurement, verbose=False, replay=False)
    
    # Check that the mean was updated and clamped (non-negative).
    assert torch.all(estimator.y_mu >= 0), "y_mu should be non-negative after update."
    # Ensure that the variance has been updated.
    assert not torch.allclose(estimator.y_var, y_var_before), "Variance should update on non-replay step."
    
    # Run a replay update; variance should remain unchanged.
    y_var_after = estimator.y_var.clone()
    estimator.diag_ekf_step(obs_model, measurement, verbose=False, replay=True)
    assert torch.allclose(estimator.y_var, y_var_after), "Variance should not update during replay."

def test_diag_ekf_step_kalman(base_estimator_info):
    """Test a single update step using the Kalman filter update format."""
    estimator = base_estimator_info
    estimator.update_format = 'kalman'
    n = 3
    obs_model = create_obs(n, update_format='kalman')
    measurement = estimator.y_mu[obs_model.state_indices] + 0.2

    # Run the update step.
    estimator.diag_ekf_step(obs_model, measurement, verbose=False, replay=False)
    
    # Check that the heterogeneous mean is updated and clamped.
    assert torch.all(estimator.y_mu >= 0), "y_mu should be clamped to non-negative values."
    # Verify that the mean has changed from the initial state.
    initial_mean = torch.tensor([0.5, 0.2, 0.8], dtype=torch.double)
    assert not torch.allclose(estimator.y_mu, initial_mean), "y_mu should change after the update."

def test_update_estimation_replay(base_estimator_info):
    """Test the update_estimation method including replay updates."""
    estimator = base_estimator_info
    estimator.update_format = 'information'
    n = 3
    obs_model = create_obs(n, update_format='information')
    
    # Simulate several observations by adding them to the replay buffer.
    for _ in range(5):
        meas = estimator.y_mu[obs_model.state_indices] + 0.15
        estimator.obs_buffer.append((obs_model, meas))
    
    mean_before = estimator.y_mu.clone()
    estimator.update_estimation(verbose=False)
    
    # Ensure that the state mean is updated.
    assert not torch.allclose(estimator.y_mu, mean_before), "y_mu should update after update_estimation."
    # Ensure that the mean is clamped to non-negative values.
    assert torch.all(estimator.y_mu >= 0), "y_mu must be non-negative after update."

def test_update_estimation_no_obs(base_estimator_info):
    """Test that update_estimation raises an error when no observations are present."""
    estimator = base_estimator_info
    estimator.obs_buffer = []  # Ensure the buffer is empty.
    with pytest.raises(AssertionError):
        estimator.update_estimation(verbose=False)

def test_nonnegative_clamping(base_estimator_info):
    """Test that the update clamps negative values to zero when non_negative is True."""
    estimator = base_estimator_info
    estimator.update_format = 'information'
    n = 3
    obs_model = create_obs(n, update_format='information')
    # Create a measurement that is significantly lower than the prediction,
    # so that the update would naturally drive the mean negative.
    measurement = estimator.y_mu[obs_model.state_indices] - 1.0
    estimator.diag_ekf_step(obs_model, measurement, verbose=False, replay=False)
    assert torch.all(estimator.y_mu >= 0), "y_mu must be clamped to non-negative values."

def test_latent_update():
    """Test that the latent state updates when a latent basis is provided."""
    max_dim = 3
    latent_dim = 2
    # Create a simple latent basis.
    A = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]], dtype=torch.double)
    x_mu = torch.tensor([0.5, 0.2, 0.8], dtype=torch.double)
    x_var = torch.tensor([1.0, 1.0, 1.0], dtype=torch.double)
    latent_mu = torch.tensor([0.1, 0.1], dtype=torch.double)
    latent_var = torch.tensor([1.0, 1.0], dtype=torch.double)
    
    estimator = DiagonalEKF(max_dim, x_mu, x_var,
                            latent_mu, latent_var,
                            latent_basis=A,
                            max_buffer_len=5,
                            num_replay_update=0,
                            non_negative=True,
                            update_format='information')
    # Override diag_AtB for testing.
    estimator.diag_AtB = diag_AtB
    obs_model = create_obs(max_dim, update_format='information')
    # Use a measurement with an offset.
    measurement = estimator.y_mu[obs_model.state_indices] + 0.3
    estimator.diag_ekf_step(obs_model, measurement, verbose=False, replay=False)
    
    # Verify that the latent mean was updated.
    assert not torch.allclose(estimator.q_mu, latent_mu), "Latent mean should update if latent is included."
    # Check that latent mean is clamped to be non-negative.
    assert torch.all(estimator.q_mu >= 0), "Latent mean must be clamped to non-negative values."
