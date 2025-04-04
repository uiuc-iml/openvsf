# tests/test_recursive_estimator.py

import pytest
import torch

from vsf.estimator.recursive_optimizer import LinearRecursiveEstimator
from vsf.estimator.recursive_optimizer import ObservationLinearization  # Example import if your code has this

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_type = {'dtype': torch.double, 'device': device}

def test_init_no_latent():
    """Test initializing LinearRecursiveEstimator without a latent basis."""
    max_dim = 5
    x_mu = 1.0
    x_var = 2.0
    
    estimator = LinearRecursiveEstimator(
        max_dim=max_dim,
        x_mu=x_mu,
        x_var=x_var,
        latent_basis=None,
        max_buffer_len=10
    )
    assert len(estimator.y_mu) == max_dim
    assert len(estimator.y_var) == max_dim
    assert torch.allclose(estimator.y_mu, torch.full((max_dim,), x_mu, **tensor_type))
    assert torch.allclose(estimator.y_var, torch.full((max_dim,), x_var, **tensor_type))
    assert estimator.A is None
    assert estimator.num_obs() == 0

def test_init_with_latent():
    """Test initializing LinearRecursiveEstimator with a latent basis."""
    max_dim = 4
    latent_dim = 2
    A = torch.rand(size=(max_dim, latent_dim), **tensor_type)
    
    estimator = LinearRecursiveEstimator(
        max_dim=max_dim,
        x_mu=0.5,
        x_var=1.0,
        latent_mu=0.1,
        latent_var=0.2,
        latent_basis=A
    )
    assert estimator.A.shape == (max_dim, latent_dim)
    assert estimator.q_mu.shape[0] == latent_dim
    assert estimator.q_var.shape[0] == latent_dim

def test_add_observation():
    """Test adding an observation to the buffer."""
    estimator = LinearRecursiveEstimator(3, 0.0, 0.1)
    obs_mat = torch.eye(3, dtype=torch.double)
    obs_bias = torch.zeros(3, dtype=torch.double)
    var = 1.0
    
    # Suppose ObservationLinearization is a simple container with:
    #   matrix, state_indices, bias, covar, etc.
    obs_lin = ObservationLinearization(matrix=obs_mat, var=var, bias=obs_bias, 
                                       state_indices=torch.arange(3))
    
    measurement = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
    estimator.add_observation(obs_lin, measurement)
    
    assert estimator.num_obs() == 1
    (stored_model, stored_measurement) = estimator.obs_buffer[0]
    assert torch.allclose(stored_model.matrix, obs_mat)
    assert torch.allclose(stored_measurement, measurement)

def test_get_mean():
    """Test mean calculation with or without latent factor."""
    # No latent factor
    estimator_no_latent = LinearRecursiveEstimator(3, 0.0, 0.1)
    mean_no_latent = estimator_no_latent.get_mean()
    # Should match y_mu directly
    assert torch.allclose(mean_no_latent, estimator_no_latent.y_mu)

    # With latent factor
    A = torch.eye(3, **tensor_type)
    estimator_latent = LinearRecursiveEstimator(
        3, 
        x_mu=1.0, 
        x_var=0.1, 
        latent_mu=2.0, 
        latent_var=0.3, 
        latent_basis=A
    )
    mean_with_latent = estimator_latent.get_mean()
    # Should be y_mu + A @ q_mu
    expected = estimator_latent.y_mu + A @ estimator_latent.q_mu
    assert torch.allclose(mean_with_latent, expected)

def test_get_var_simple_basis():
    """Test diagonal variance computation."""
    estimator = LinearRecursiveEstimator(3, 0.0, 0.1)
    var_no_latent = estimator.get_var()
    # Should match y_var when no latent factor
    assert torch.allclose(var_no_latent, estimator.y_var)

    A = torch.eye(3, **tensor_type)
    est_latent = LinearRecursiveEstimator(
        3, 
        x_mu=0.0, 
        x_var=0.1,
        latent_mu=0.5, 
        latent_var=0.2, 
        latent_basis=A
    )
    var_with_latent = est_latent.get_var()
    # Should be y_var + (A @ q_var) * A, which in the case of an identity basis 
    # is simply y_var + q_var
    expected = est_latent.y_var + est_latent.q_var
    assert torch.allclose(var_with_latent, expected), f"Expected: {expected}, got: {var_with_latent}"

def test_get_var_random_basis():
    """Test variance computation with a random latent basis."""
    estimator = LinearRecursiveEstimator(3, 0.0, 0.1)
    var_no_latent = estimator.get_var()
    # Should match y_var when no latent factor
    assert torch.allclose(var_no_latent, estimator.y_var)

    for i in range(10):
        A = torch.rand(3, 3, **tensor_type)
        est_latent = LinearRecursiveEstimator(
            3, 
            x_mu=0.0, 
            x_var=0.1,
            latent_mu=0.5, 
            latent_var=0.2, 
            latent_basis=A
        )
        var_with_latent = est_latent.get_var()
        # Correct but inefficient way to compute the variance with latent factor
        latent_var = (A @ torch.diag(est_latent.q_var) @ A.T).diag()
        expected = est_latent.y_var + latent_var
        assert torch.allclose(var_with_latent, expected), f"Expected: {expected}, got: {var_with_latent}"

def test_not_implemented_update():
    """Test that update_estimation is not implemented."""
    estimator = LinearRecursiveEstimator(3, 1.0, 0.5)
    with pytest.raises(NotImplementedError):
        estimator.update_estimation()

def test_state_dict_loading():
    """Test saving and loading the state dict."""
    est = LinearRecursiveEstimator(3, x_mu=1.0, x_var=2.0)
    saved = est.state_dict()

    new_est = LinearRecursiveEstimator(3, x_mu=0.0, x_var=0.0)
    new_est.load_state_dict(saved)

    assert torch.allclose(new_est.y_mu, est.y_mu)
    assert torch.allclose(new_est.y_var, est.y_var)


