import numpy as np
import torch
from typing import List, Union, Optional
import cvxpy as cp
import time
from ..utils.data_utils import convert_to_tensor, convert_to_numpy
from .recursive_optimizer import ObservationLinearization, LinearRecursiveEstimator


def gaussian_nll(x: cp.Variable, mu: np.ndarray = None, lam: np.ndarray = None, constant=False) -> cp.Expression:
    """
    Computes the Gaussian negative log likelihood for a CVXPY variable.

    The Gaussian is defined by the mean `mu` and the precision matrix `lam`.
    We calculate the negative log-likelihood of the Gaussian prior,
    evaluated at the value of `x`.

    Args:
    - x (cp.Variable): The location to evaluate.
    - mu (np.ndarray): Mean of the Gaussian prior.
    - lam (np.ndarray): Precision matrix of the Gaussian prior.
    - constant (bool) whether to include the constant offset 
    """

    if constant:
        raise NotImplementedError("TODO: NLL constants?")
    if (mu is not None) and (lam is not None):
        if lam.ndim == 1:
            res_squares = cp.power(x - mu, 2)
            return 0.5*cp.multiply(res_squares, lam).sum()
        else:
            return 0.5*cp.quad_form(x - mu, cp.psd_wrap(lam))
    else:
        return 0.0
    

class QuadProgOptimizer(LinearRecursiveEstimator):
    """
    Estimator that uses quadratic programming to solve for a linear estimation
    problem.

    This estimator assumes a set of linear observations are made on the state x,
    which are corrupted by Gaussian noise. The maximum likelihood estimate of x is
    obtained by solving a quadratic programming problem.

    The state x is composed of a heterogeneous factor y and latent factor q, both
    of which are optional:

        x = y + A*q in R^n 
        
    where y in R^n is a heterogeneous term and q in R^k is a latent factor. 
    If the latent factor is included, A is a known basis matrix.  Diagonal priors 
    should be defined over y and q,
    
        y ~ N(mu_y,diag(Sig_y)),      q ~ N(mu_q,diag(Sig_q))

    such that x has the prior N(mu_y + A*mu_q, diag(Sig_y) + A * diag(Sig_q) * A^T).
    To mark the heterogeneous factor as optional, set Sig_y = 0.  To mark the
    latent factor as optional, set Sig_q = 0 or A = None.
        
    Observations are given by vectors z^i, observation matrices W^i, optional
    indices ind^i, biases z0^i, and covariances Sig_z^i such that
    
        z^i = W^i * x[ind^i] + z0^i + eps^i,     eps^i ~ N(0,Sig_z^i).

    It is assumed that both y and q are non-negative.  The optimization problem
    is formulated as a quadratic program with linear constraints.  The objective
    is to maximize the likelihood of the state given the observations and prior.
    """

    def __init__(self, max_dim:int,
                 x_mu:Union[float,torch.Tensor]=0.0, x_var:Union[float,torch.Tensor]=0.0,
                 latent_mu:Union[float,torch.Tensor]=0.0, latent_var:Union[float,torch.Tensor]=0.0,
                 latent_basis:Optional[torch.Tensor]=None,
                 max_buffer_len:int=None) -> None:
        """
        Initialize the estimator with standard LinearRecursiveEstimator parameters."""
        super().__init__(max_dim,x_mu,x_var,latent_mu,latent_var,latent_basis,max_buffer_len)
        self.include_heterogeneous = (isinstance(x_var,torch.Tensor) or x_var != 0.0)
        self.include_latent = self.A is not None and (isinstance(latent_var,torch.Tensor) or latent_var != 0.0)
        if self.include_heterogeneous:
            self.y_lam  = 1.0 / self.y_var 
        if self.include_latent:
            self.q_lam = 1.0 / self.q_lam

        self.y_variable: cp.Variable = None
        self.q_variable: cp.Variable = None
        self.x_variable: cp.Variable = None
        self.A_subset = None

    def setup_opt_var(self, obs_idx: np.ndarray, verbose=False):
        """
        Setup the CVXPY optimization variables.

        Args:
        - obs_idx: The indices of the stiffness values to be estimated.
        - verbose: Flag to print the information during the setup.
        """
        K_dim = obs_idx.shape[0]

        if self.include_heterogeneous:
            self.y_variable = cp.Variable((K_dim,))
        if self.include_latent:
            self.q_variable = cp.Variable((self.num_basis,))
            self.A_subset = self.A[obs_idx, :]
        if self.include_heterogeneous and self.include_latent:
            self.x_variable = self.y_variable + self.A_subset @ self.q_variable
        elif self.include_heterogeneous:
            self.x_variable = self.y_variable
        else:
            self.x_variable = self.A_subset.dot(self.q_variable)

    def solve(self, unified_x_indices: np.ndarray, obs_models:List[ObservationLinearization], measurements:List[torch.Tensor],
              reg_weight: float = 1.0, solver='OSQP', verbose=False):
        """
        Solve a quadratic programming optimization problem to 
        get optimal estimate of y_mu and q_mu.

        The estimation process involves:
        1. Setting up CVXPY optimization variables.
        2. Computing the observation loss as a cp.Expression.
        3. Computing the regularization loss as cp.Expression.
        4. Solving the optimization problem using the specified `solver`.

        Args:
        - unified_x_indices (np.ndarray): Indices of the x variable to be estimated. All indices in the observation models must be a subset of these
        - obs_models (list[ObservationLinearization]): List of observation models.
        - measurements (list[Tensor]): List of actual observations
        - reg_weight (float): Weight for the regularization term.  Should be 1 if your priors are set correctly.
        - solver (str): Solver used for the optimization problem (default: 'OSQP').
        - verbose (bool): If True, prints additional information during optimization.

        Updates:
            The estimated value of `y_mu`, `q_mu`.
        """
        self.setup_opt_var(unified_x_indices, verbose=verbose)

        constraints = []
        obs_loss = 0.0
        for obs_model, meas in zip(obs_models, measurements):
            obs_mat = obs_model.matrix.cpu().numpy()
            if obs_model.state_indices is not None:
                pred = obs_mat @ self.x_variable[obs_model.state_indices.cpu().numpy()]
            else:
                pred = obs_mat @ self.x_variable[:]
            assert pred.ndim == 1
            assert pred.shape[0] == len(meas)
            if obs_model.bias is not None:
                pred += obs_model.bias.cpu().numpy()
            assert pred.ndim == 1
            assert pred.shape[0] == len(meas)
            assert len(meas) == len(obs_model.var)
            obs_loss += gaussian_nll(pred, meas.cpu().numpy(), obs_model.var.cpu().numpy())

        reg_loss = 0
        if self.include_heterogeneous:
            constraints.append(self.y_variable >= 0)
            assert self.y_variable.ndim == 1
            assert self.y_variable.shape[0] == len(unified_x_indices)
            assert self.y_mu[unified_x_indices].ndim == 1
            assert self.y_lam[unified_x_indices].ndim == 1
            reg_loss += gaussian_nll(self.y_variable, self.y_mu[unified_x_indices], self.y_lam[unified_x_indices])
        if self.include_latent:
            constraints.append(self.q_variable >= 0)
            reg_loss += gaussian_nll(self.q_variable, self.q_mu, self.q_lam)
        
        prob = cp.Problem(
            cp.Minimize(
                obs_loss +
                reg_weight *
                reg_loss),
            constraints)

        print("QuadProgOptimizer: start solving cvxpy problem...")
        start_time = time.time()
        prob.solve(verbose=verbose, solver=solver)
        solve_time = time.time() - start_time
        print(f"QuadProgOptimizer: cvxpy solve time = {solve_time}")

        # Update variables
        if self.include_heterogeneous:
            self.y_mu[unified_x_indices] = torch.from_numpy(self.y_variable.value).to(self.y_mu.device).to(self.y_mu.dtype)
        if self.include_latent:
            self.q_mu[:] = torch.from_numpy(self.q_variable.value).to(self.q_mu.device).to(self.q_mu.dtype)
        
    def update_estimation(self):
        """Note: this estimator is non recursive"""
        import warnings
        warnings.warn("QuadProgOptimizer does not operate in recursive form")
    
    def finalize_estimation(self):
        """Finalize the estimation after all observations are added.
        
        Converts all observation indices into a subset of indices.  Then
        calls solve()
        """
        all_observed_indices = self.get_observed_indices()
        print("Observed indices:",len(all_observed_indices))
        #reshape all observation indices
        observed_to_idx = torch.full((len(self.y_mu),),-1,dtype=int,device=all_observed_indices.device)
        observed_to_idx[all_observed_indices] = torch.arange(0,len(all_observed_indices),dtype=int,device=all_observed_indices.device)
        measurements = []
        new_obs_models = []
        for (obs_model,meas) in self.obs_buffer:
            new_indices = all_observed_indices if obs_model.state_indices is None else observed_to_idx[obs_model.state_indices]
            new_obs_models.append(ObservationLinearization(matrix=obs_model.matrix,var=obs_model.var,bias=obs_model.bias,
                                                           state_indices=new_indices))
            measurements.append(meas)
        self.solve(all_observed_indices, new_obs_models, measurements)