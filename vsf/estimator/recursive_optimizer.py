import os
import pickle
import time
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import cvxpy as cp
import torch
import scipy
from dataclasses import dataclass


@dataclass
class ObservationLinearization:
    """Stores an observation model for a LinearRecursiveEstimator.

    Model is measurement = matrix * x[state_indices] + bias + eps
    with eps ~ N(0,var).
    """
    matrix : torch.Tensor                   # shape #obs x #indices or #obs x n if state_indices is None
    var : torch.Tensor                      # shape #obs or #obs x #obs
    bias : Optional[torch.Tensor] = None    # shape #obs
    state_indices : Optional[torch.Tensor] = None   # shape #indices
    
    def __post_init__(self):
        assert self.matrix.ndim == 2
        nobs = self.matrix.size(0)
        if self.state_indices is not None:
            assert self.state_indices.ndim == 1
            assert self.state_indices.size(0) == self.matrix.size(1)
        if self.bias is not None:
            assert self.bias.size(0) == nobs
        if isinstance(self.var,(float,int)):
            self.var = torch.full((nobs,),self.var,dtype=self.matrix.dtype,device=self.matrix.device)
        if self.var.ndim == 1:
            assert self.var.size(0) == nobs
        elif self.var.ndim == 2:
            assert self.var.size(0) == nobs
            assert self.var.size(1) == nobs
        else:
            raise ValueError("Invalid var shape")
    
    def diag_var(self):
        """Returns the diagonal of the covariance matrix"""
        if self.var.ndim == 1:
            return self.var
        return torch.diag(self.var)

    def covar(self):
        """Returns the full covariance matrix"""
        if self.var.ndim == 1:
            return torch.diag(self.var)
        return self.var
    
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        """Standard mean prediction.  x is the state vector"""
        if self.state_indices is not None:
            pred = self.matrix @ x[self.state_indices]
        else:
            pred = self.matrix @ x
        if self.bias is not None:
            return pred + self.bias
        return pred
    
    def predict_batch(self, xs:torch.Tensor) -> torch.Tensor:
        """Batch mean prediction. xs is a B x N state tensor batch."""
        if self.state_indices is not None:
            pred = xs[:,self.state_indices] @ self.matrix.T
        else:
            pred = xs @ self.matrix.T
        if self.bias is not None:
            return pred + self.bias
        return pred
    
    def to(self, device, dtype=None):
        """
        Move the model to the given device and dtype.
        
        NOTE: this will not create a copy of the model, but only
        internally change the device and dtype of the tensors.
        """
        self.matrix.to(device)
        self.var.to(device)
        if dtype is not None:
            self.matrix = self.matrix.to(dtype)
            self.var = self.var.to(dtype)
        if self.bias is not None:
            self.bias.to(device)
            if dtype is not None:
                self.bias = self.bias.to(dtype)
        if self.state_indices is not None:
            self.state_indices.to(device)
            if dtype is not None:
                self.state_indices = self.state_indices.to(dtype)
        return self
    
    @staticmethod
    def merge(*obs_models : 'ObservationLinearization') -> 'ObservationLinearization':
        """
        Merge multiple sparse linear observation models into one obsernation model.

        This method takes a list of `ObservationLinearization` objects, each of
        which may observe only a subset of the full state, the return will have 
        a single merged observation model with the following properties: 
            
            matrix: single merged observation matrix of shape
            (sum_i n_obs_i, n_unique_state_indices),
            bias: concatenated bias vector of shape (sum_i n_obs_i,),
            var: concatenated variance vector of shape (sum_i n_obs_i,),
            state_indices: 1D array of the sorted unique state indices.

        Internally, it uses `np.unique(..., return_inverse=True)` on the
        concatenated state index lists to compute the overall state dimension
        and to map each submatrix into the correct columns of the merged
        matrix.

        Args:

            obs_models : List[ObservationLinearization]
                List of sparse observation models to merge. Each must have
                `.matrix` (Tensor), `.var` (Tensor or scalar), optional
                `.bias` (Tensor), and optional `.state_indices` (1D Tensor).
                If `state_indices is None`, the model is assumed to act on the
                full state vector of size `matrix.size(1)`.

        Returns:

            merged_obs : ObservationLinearization
                A new observation model whose
                - `merged_obs.matrix` is shape `(total_meas, n_unique)`,
                - `merged_obs.bias`  is shape `(total_meas,)`,
                - `merged_obs.var`   is shape `(total_meas,)` (variances),
                - `merged_obs.state_indices` is a 1D LongTensor of length `n_unique`.
                `merged_obs.state_indices[i]` gives the original state index
                corresponding to column `i` of `merged_obs.matrix`.
            merged_measurement: np.ndarray
                An array of shape `(total_meas,)` containing the concatenated
                measurements from all models. The order of the measurements
                corresponds to the order of the rows in `merged_obs.matrix`,
                which is the order in the measurement_list.

        """
        # 0) convert to list if a single model is passed
        obs_models = list(obs_models)
        
        # 1) how many rows each model contributes
        num_meas_list = [model.matrix.shape[0] for model in obs_models]
        total_meas = sum(num_meas_list)
        
        tsr_params = { 'dtype': obs_models[0].matrix.dtype, 
                       'device': obs_models[0].matrix.device }

        # 2) gather all state_indices as one long array
        state_idx_lists = [
            (obs.state_indices.numpy() if obs.state_indices is not None
            else np.arange(obs.matrix.size(1)))
            for obs in obs_models
        ]
        all_state_idxs = np.concatenate(state_idx_lists)

        # compute unique indices + inverse map
        unique_indices, inverse = np.unique(all_state_idxs, return_inverse=True)
        # split inverse back per-model
        splits = np.cumsum([len(idx) for idx in state_idx_lists])[:-1]
        per_model_inverse = np.split(inverse, splits)

        # 3) build merged observation matrix
        merged_matrix = torch.zeros((total_meas, len(unique_indices)), **tsr_params)
        
        # row block boundaries
        row_ends = np.cumsum(num_meas_list)
        row_starts = np.concatenate([[0], row_ends[:-1]])

        for (obs, r0, r1, inv_idx) in zip(
            obs_models, row_starts, row_ends, per_model_inverse
        ):
            # obs.matrix is shape (n_obs_i, len(inv_idx))
            merged_matrix[r0:r1, inv_idx] = obs.matrix
        
        # 4) merge biases (fill with zeros if None)
        merged_bias = torch.cat([
            (obs.bias if obs.bias is not None else torch.zeros(n, **tsr_params))
            for obs, n in zip(obs_models, num_meas_list) ], dim=0)

        # 5) merge variances as vector of diag elements
        # TODO: support merging full covariance matrices
        merged_var = torch.cat([obs.diag_var() for obs in obs_models], dim=0)

        # wrap in a new ObservationLinearization
        merged_state_tensor = torch.tensor(unique_indices,
                                           device=tsr_params['device'],
                                           dtype=torch.long)
        merged_obs = ObservationLinearization(
            matrix=merged_matrix,
            var=merged_var,
            bias=merged_bias,
            state_indices=merged_state_tensor
        )

        return merged_obs


def diag_AtB(A : torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Returns the diagonal of A.T @ B with A and B both m x n matrices"""
    assert A.shape == B.shape
    return (A * B).sum(dim=0) 


class LinearRecursiveEstimator:
    """Base estimator for a high-dimensional stationary state with observations
    as linear-Gaussian functions of state.

    The state x is composed of a heterogeneous factor y and latent factor q, both
    of which are optional:
    
    .. math::

        x = y + A*q in R^n 
        
    where y in R^n is a heterogeneous term and q in R^k is a latent factor. 
    If the latent factor is included, A is a known basis matrix.  Diagonal priors 
    should be defined over y and q,
    
    .. math::
    
        y ~ N(mu_y,diag(Sig_y)),      q ~ N(mu_q,diag(Sig_q))

    such that x has the prior N(mu_y + A*mu_q, diag(Sig_y) + A * diag(Sig_q) * A^T).
    To mark the heterogeneous factor as optional, set Sig_y = 0.  To mark the
    latent factor as optional, set Sig_q = 0 or A = None.
        
    Observations are given by vectors z^i, observation matrices W^i, optional
    indices ind^i, biases z0^i, and covariances Sig_z^i such that
    
    .. math::
    
        z^i = W^i * x[ind^i] + z0^i + eps^i,     eps^i ~ N(0,Sig_z^i).

    This class provides shared methods for different instantiations of the
    estimator.  It gives a replay buffer and 

    Unified interface:
    
    1. **add_observation(obs_model, measurement)**  
       Adds an observation model and measurement to the buffer.

    2. **update_estimation()**  
       Reads the most recent measurements. This function is implementation-dependent.

    3. **finalize_estimate()**  
       Updates `y_mu`, `y_var`, `q_mu`, `q_var` so the `x` mean/variance can be extracted.  
       This function is implementation-dependent.

       
    NOTE: for numerical stability purposes, all torch tensors in this class are double precision.

    Attributes:
        obs_buffer: the observation buffer, stores (observation model, measurement) pairs
        max_buffer_len: the max length of the observation buffer, or None for unlimited buffer. 
        y_mu, y_var: the mean and variance of the heterogeneous factor y
        q_mu, q_var: the mean and variance of the latent factor q
        A: the basis matrix for the latent factor q
    """
    def __init__(self, max_dim:int,
                 x_mu:Union[float,torch.Tensor]=0.0, x_var:Union[float,torch.Tensor]=0.0,
                 latent_mu:Union[float,torch.Tensor]=0.0, latent_var:Union[float,torch.Tensor]=0.0,
                 latent_basis:Optional[torch.Tensor]=None,
                 max_buffer_len:Optional[int] = None) -> None:
        self.max_buffer_len = max_buffer_len
        self.obs_buffer = []    # type : List[Tuple[ObservationLinearization,torch.Tensor]] 
        
        if isinstance(x_mu,torch.Tensor):
            device = x_mu.device
        else:
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
        # NOTE: all tensors are double precision for numerical stability
        self.tsr_params = {'dtype': torch.double, 'device': device}

        if isinstance(x_mu,torch.Tensor):
            assert x_var.ndim == 1,"Mean must be a vector"
            self.y_mu = x_mu.to(**self.tsr_params)
        else:
            self.y_mu : torch.Tensor = torch.full((max_dim,), x_mu, **self.tsr_params)
        if isinstance(x_var,torch.Tensor):
            assert x_var.ndim == 1,"Variance must be a per-point variance"
            self.y_var = x_var.to(**self.tsr_params)
        else:
            self.y_var : torch.Tensor = torch.full((max_dim,), x_var, **self.tsr_params)
        assert len(self.y_mu) == len(self.y_var)
        
        # create the latent basis matrix 
        if latent_basis is not None:
            assert latent_basis.ndim == 2
            assert latent_basis.shape[0] == max_dim
        self.A : torch.Tensor = latent_basis
        if self.A is not None:
            self.A = self.A.to(**self.tsr_params)
        latent_dim = 0 if latent_basis is None else latent_basis.shape[1]
        if isinstance(latent_mu,torch.Tensor):
            assert latent_var.ndim == 1,"Latent mean must be a vector"
            self.q_mu = latent_mu.to(**self.tsr_params)
        else:
            self.q_mu : torch.Tensor = torch.full((latent_dim,), latent_mu, **self.tsr_params)
        assert self.q_mu.size(0) == latent_dim
        if isinstance(latent_var,torch.Tensor):
            assert latent_var.ndim == 1,"Latent variance must be a vector"
            self.q_var = latent_var.to(**self.tsr_params)
        else:
            self.q_var : torch.Tensor = torch.full((latent_dim,), latent_var, **self.tsr_params)
        assert self.q_var.size(0) == latent_dim
        

    def to(self, device):
        self.y_mu.to(device)
        self.y_var.to(device)
        self.q_mu.to(device)
        self.q_var.to(device)
        if self.A is not None:
            self.A.to(device)
        return self

    def num_obs(self):
        """Returns the number of observations in the buffer."""
        return len(self.obs_buffer)

    def clear_obs_buffer(self):
        """Clear the observation buffer, set self.obs_buffer to an empty list."""
        self.obs_buffer = []

    def add_observation(self, model : ObservationLinearization, measurement : torch.Tensor):
        """Add an observation to the buffer."""
        assert isinstance(measurement,torch.Tensor)
        # estimator only supports 1D measurements 
        # so we flatten the measurement tensor
        if measurement.ndim > 1:
            measurement = measurement.flatten()
        assert measurement.device == model.matrix.device,'Inconsistent tensor devices in observation'
        self.obs_buffer.append((model,measurement))

        if self.max_buffer_len is not None and self.num_obs() > self.max_buffer_len: # fifo buffer
            self.obs_buffer.pop(0)
    
    def get_mean(self, idx=None) -> torch.Tensor:
        """Returns the estimated x mean."""
        if idx is None:
            if self.A is not None:
                return self.y_mu + self.A @ self.q_mu
            else:
                return self.y_mu.clone()
        else:    
            x = self.y_mu[idx]
            if self.A is not None:
                return x + self.A[idx, :] @ self.q_mu
            else:
                return x.clone()

    def get_var(self, idx=None) -> torch.Tensor:
        """Returns the diagonal variance of x."""
        # Return the full covariance matrix if idx is None
        if idx is None:
            if self.A is not None:
                return self.y_var + (self.A**2) @ self.q_var
            else:
                return self.y_var.clone()
        # Return the diagonal variance of the selected indices
        else:
            var = self.y_var[idx]
            if self.A is not None:
                A = self.A[idx, :]
                return var + (A**2) @ self.q_var
            else:
                return var.clone()
    
    def get_covar(self, idx=None) -> torch.Tensor:
        """Returns the full covariance matrix of x."""
        if idx is None:
            if self.A is not None:
                return torch.diag(self.y_var) + self.A @ torch.diag(self.q_var) @ self.A.T
            else:
                return torch.diag(self.y_var)
        else:
            cov = torch.diag(self.y_var[idx])
            if self.A is not None:
                A = self.A[idx, :]
                return cov + A @ torch.diag(self.q_var) @ A.T
            else:
                return cov

    def update_estimation(self):
        """Update the estimation based on the current observation buffer"""
        raise NotImplementedError
    
    def finalize_estimation(self):
        """Finalize the estimation after all observations are added"""
        pass
    
    def predict_observation(self, obs : ObservationLinearization) -> torch.Tensor:
        """Predicts the mean observation at the current state."""
        res = obs.matrix @ self.get_mean(idx=obs.state_indices)
        if obs.bias is not None:
            return res + obs.bias
        return res
        
    def get_observed_indices(self) -> torch.LongTensor:
        """Returns all the indices observed across all observations in the buffer """
        touch_idx = torch.concatenate([o.state_indices for o,m in self.obs_buffer])
        return torch.unique(touch_idx)
    
    def get_unobserved_indices(self) -> torch.LongTensor:
        """Complement of get_observed_indices. 
        
        NOTE: converts to CPU
        """
        touch_idx = self.get_observed_indices()
        return torch.tensor(np.setdiff1d(np.arange(len(self.x), touch_idx.cpu().numpy())),device=touch_idx.device,dtype=touch_idx.dtype)

    def state_dict(self):
        """Subclasses may override this to save additional state."""
        res = {'y_mu': self.y_mu, 'y_var': self.y_var, 'q_mu': self.q_mu, 'q_var': self.q_var}
        if self.A is not None:
            res['A'] = self.A
        return res
    
    def load_state_dict(self, state_dict : dict):
        """Subclasses may override this to load additional state."""
        self.y_mu = state_dict['y_mu']
        self.y_var = state_dict['y_var']
        self.q_mu = state_dict['q_mu']
        self.q_var = state_dict['q_var']
        if 'A' in state_dict:
            self.A = state_dict['A']


class SGDEstimator(LinearRecursiveEstimator):
    """
    Stochastic Gradient Descent (SGD) estimator for linear estimation problem.

    Uses a replay buffer with the given batch_size for each update step.

    If you don't use finalize_estimation(), you will need to detach the mean
    / variance tensors before using them.
    """
    def __init__(self, max_dim:int,
                 x_mu:Union[float,torch.Tensor]=0.0, x_var:Union[float,torch.Tensor]=0.0,
                 latent_mu:Union[float,torch.Tensor]=0.0, latent_var:Union[float,torch.Tensor]=0.0,
                 latent_basis:Optional[torch.Tensor]=None,
                 max_buffer_len:int = None, 
                 non_negative:bool=True, batch_size:int=100) -> None:
        super().__init__(max_dim, x_mu,x_var,latent_mu,latent_var,latent_basis,max_buffer_len)

        include_heterogeneous = (isinstance(x_var,torch.Tensor) or x_var != 0.0)
        include_latent = self.A is not None and (isinstance(latent_var,torch.Tensor) or latent_var != 0.0)
        
        if include_heterogeneous:
            params_lst = [self.y_mu]
        else:
            params_lst = []
        if include_latent:
            params_lst += [self.q_mu]
        for p in params_lst:
            p.requires_grad_()
        self.optimizer = torch.optim.Adam(params_lst, lr=1e-3)

        self.observation_loss = torch.nn.GaussianNLLLoss(full=True)
        self.regularization_loss = torch.nn.GaussianNLLLoss(full=True)
        self.batch_size = batch_size
        self.non_negative = non_negative
        
    def sgd_step(self, obs_model : ObservationLinearization, measurement : torch.Tensor) -> float:
        """
        Rune a single stochastic gradient descent step on the given observation.
        """
        tau_hat = self.predict_observation(obs_model)
        loss = self.observation_loss(measurement, tau_hat, obs_model.var) 
        loss += self.regularization_loss(self.y_mu, torch.zeros_like(self.y_mu), self.y_var) 
        if self.A is not None:
            loss += self.regularization_loss(self.q_mu, torch.zeros_like(self.q_mu), self.q_var)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_estimation(self, verbose=False):
        """
        SGD update step on the observation buffer.
        
        The estimator will sample a batch of observations from the buffer and
        run a single SGD step on each observation.
        """
        self.q_mu.requires_grad_(True)
        self.y_mu.requires_grad_(True)
        num_sample = min(self.batch_size, self.num_obs())
        step_idx_ary = np.random.randint(0, self.num_obs(), num_sample)
        for step_idx in step_idx_ary:
            loss = self.sgd_step(*self.obs_buffer[step_idx])
            if verbose:
                print("loss:", loss)
        
        if self.non_negative:
            with torch.no_grad():
                self.y_mu.clamp_(min=0)
                if self.A is not None:
                    self.q_mu.clamp_(min=0)
    
    def finalize_estimation(self):
        """
        SGD estimator does not change estimation during finalization.
        
        This function will disable gradient computation on the mean tensors 
        to support evaluation steps.
        """
        self.y_mu.requires_grad_(False)
        self.q_mu.requires_grad_(False)
    


class DiagonalEKF(LinearRecursiveEstimator):
    """
    Diagonal EKF estimator that maintains the diagonal part of the
    covariance matrix for the estimation.

    Note: this will use the information filter formulation with *_lam indicating
    the inverse of the diagonal of the covariance matrix (diagonal precision),
    and *_mu being the product of the precision matrix and the normal mean.
    
    On the state x=(y,q), where y is heterogeneous and q is latent, the
    observation model is z = W*Sel[ind]*[I | A]*x + eps 
    which implies the observation matrix is H = W*[Sel[ind] | A[ind:]].

    TODO: math below may be incorrect, need to verify
    Hence, the information filter update computes 
    delta_i = H^T * R^-1 z 
            = [ Sel[ind]^T W^T * R^-1 * z \\
                A[ind:]^T W^T * R^-1 * z ] 
    and 
    variance I = H^T R^-1 H 
               = [Sel[ind]^T] W^T * R^-1 * W [Sel[ind] | A[ind:]] [A[ind:]^T ]
    And then updates (y_lam, q_lam) += diag(I), (y_mu, q_mu) += i. 
    """

    def __init__(self, max_dim:int,
                 x_mu:Union[float,torch.Tensor]=0.0, x_var:Union[float,torch.Tensor]=0.0,
                 latent_mu:Union[float,torch.Tensor]=0.0, latent_var:Union[float,torch.Tensor]=0.0,
                 latent_basis:Optional[torch.Tensor]=None,
                 max_buffer_len:int = None, num_replay_update:int=10, 
                 non_negative:bool=True, update_format='kalman') -> None:
        super().__init__(max_dim, x_mu, x_var, latent_mu, latent_var, latent_basis, max_buffer_len)
        self.include_heterogeneous = (isinstance(x_var,torch.Tensor) or x_var != 0.0)
        self.include_latent = self.A is not None and (isinstance(latent_var,torch.Tensor) or latent_var != 0.0)
        self.num_replay_update = num_replay_update
        self.non_negative = non_negative
        # NOTE: Diagonal EKF has DIFFERENT update results using Kalman or Information filter formulations
        #       The Kalman update is more numerically stable, 
        #       but the Information filter update does not need to invert the covariance matrix.
        assert update_format in ['kalman','information']
        self.update_format = update_format
    
    def diag_ekf_step(self, obs_model : ObservationLinearization, 
                      measurement : torch.Tensor, verbose:bool=False, replay=False):
        """
        Run a single Diagonal EKF update step on the given observation.
        
        If replay is True, the estimator will not update the variance terms.
        
        Args:
            obs_model: a linear observation model
            measurement: the observation value as a torch tensor
            verbose: whether to print debug information
            replay: whether the update is triggered by a replay
        """
        if verbose:
            start_time = time.time()

        obs_idx = obs_model.state_indices
        obs_W_tsr = obs_model.matrix
        if verbose:
            print("obs tau tsr:", measurement)
            print('obs_W_tsr shape:', obs_W_tsr.T.shape)
        
        # compute the residual error in the observation
        z = measurement.clone()
        if obs_model.bias is not None:
            z -= obs_model.bias
        z_hat = obs_model.predict(self.get_mean())
        res_z = z - z_hat
        
        # Compute inverse of the covariance matrix in observation noise
        if obs_model.var.ndim == 1:
            # obs_model.var is a scalar
            Rinv = 1.0 / obs_model.var
            Rinvz = Rinv * res_z
        else:
            Rinv = torch.linalg.inv(obs_model.var)
            Rinvz = Rinv @ res_z
        if verbose:
            print("Rinv shape:", Rinv.shape)
            print("Rinvz shape:", Rinvz.shape)
            
        # NOTE: Very important!!! The covariance update should happen BEFORE
        #       the mean update, as the mean update uses the covariance matrix
        #       at the current time step.
        #       Want the diagonal of these fat (n > m) matrices
        #       Wt_Rinv_W = (obs_W_tsr.T @ Rinv * obs_W_tsr) if obs_model.var.ndim == 1 else (obs_W_tsr.T @ Rinv) @ obs_W_tsr
        #       and this self.A[obs_idx, :].T @ Wt_Rinv_W @ self.A[obs_idx, :]
        if not replay:
            if self.include_heterogeneous:
                # Compute information matrix update for the heterogeneous factor
                if obs_model.var.ndim == 1:
                    y_lam_inc = diag_AtB(obs_W_tsr, Rinv[:,None] * obs_W_tsr)
                else:
                    y_lam_inc = diag_AtB(obs_W_tsr, Rinv @ obs_W_tsr)
                
                y_lam_updated = 1.0 / self.y_var[obs_idx] + y_lam_inc
                self.y_var[obs_idx] = 1.0 / y_lam_updated
            if self.include_latent:
                WA = obs_W_tsr @ self.A[obs_idx,:]
                self.q_var += Rinv @ diag_AtB(WA,WA) if obs_model.var.ndim == 1 else diag_AtB(WA, Rinv @ WA)
                
        # Update the mean of the state
        # first get diagonal component of y_var
        y_var = self.get_var(obs_idx)
        if verbose:
            print('y_var min, mean, max:', y_var.min(), y_var.mean(), y_var.max())
        # Information filter update
        if self.update_format == 'information':
            y_mu_inc = y_var * (obs_W_tsr.T @ Rinvz)
        # Kalman filter update
        elif self.update_format == 'kalman':
            sig_WT = y_var.unsqueeze(1) * obs_W_tsr.T
            if obs_model.var.ndim == 1:
                R_mat = torch.diag(obs_model.var)
            else:
                R_mat = obs_model.var
            W_sig_WT = obs_W_tsr @ sig_WT
            y_mu_inc = sig_WT @ torch.linalg.inv(W_sig_WT + R_mat) @ res_z
        if self.include_heterogeneous:
            self.y_mu[obs_idx] += y_mu_inc
        if self.include_latent:
            # TODO: fix latent mean update rules
            q_mu_inc = (self.A[obs_idx, :].T @ obs_W_tsr @ Rinvz)
            self.q_mu += q_mu_inc

        if verbose:
            print("number of points:", obs_idx.shape[0])
            print("update step time:", time.time()-start_time)

    def update_estimation(self, verbose:bool=False):
        """
        Update estimation of Diagonal EKF estimator.
        
        If reply is False, the estimator will only update 
        using the most recent observation. 
        
        If replay is True, the estimator will first update
        using the most recent observation, then replay a fixed
        number of history observations in the replay buffer.
        """
        assert self.num_obs() > 0, "Need to have at least one observation"
        
        #do the update
        self.diag_ekf_step(*self.obs_buffer[-1], verbose, replay=False)
            
        if self.num_replay_update > 0:
            step_idx_ary = np.random.randint(0, self.num_obs(), 
                                             (self.num_replay_update,))
            for step_idx in step_idx_ary:
                self.diag_ekf_step(*self.obs_buffer[step_idx], verbose, replay=True)

        #clamp
        if self.non_negative:
            with torch.no_grad():
                self.y_mu.clamp_(min=0)
                if self.A is not None:
                    self.q_mu.clamp_(min=0)
    

class DenseEKF(LinearRecursiveEstimator):
    """
    Dense EKF estimator that maintains the full dense covariance
    matrix for the estimation.
    
    This will try to maintain the full dense covariance matrix for only
    the observed indices, and will extend the matrix as needed. 

    finalize_estimation() is essential here to get the updates.

    Attributes:
        iter_per_update: number of iteration per update
        len_in_mem: length of the memory
        mem_inc: memory increase per update
        idx_raw2obs: mapping from raw index to observed state index
        info_mat: subset dense information matrix for (q,y) covariance
        update_method: update method for the estimation
        non_negative: whether the estimation is non-negative
        num_replay_update: how many replays to use in each update 
    """
    def __init__(self, max_dim:int,
                 x_mu:Union[float,torch.Tensor]=0.0, x_var:Union[float,torch.Tensor]=0.0,
                 latent_mu:Union[float,torch.Tensor]=0.0, latent_var:Union[float,torch.Tensor]=0.0,
                 latent_basis:Optional[torch.Tensor]=None,
                 max_buffer_len:int = None, init_len_in_mem: int=100, 
                 update_method: str='inv', num_replay_update : int=0, non_negative: bool = True) -> None:
        super().__init__(max_dim, x_mu, x_var, latent_mu, latent_var, latent_basis, max_buffer_len)
        self.include_heterogeneous = (isinstance(x_var,torch.Tensor) or x_var != 0.0)
        self.include_latent = self.A is not None and (isinstance(latent_var,torch.Tensor) or latent_var != 0.0)
        
        self.idx_raw2obs = -1*torch.ones(max_dim, device = self.y_mu.device, dtype=int)

        self.iter_per_update = 1
        self.mem_inc = 1000

        self.num_latent = 0
        self.info_mat = torch.eye(init_len_in_mem,device=self.y_mu.device, dtype=self.y_mu.dtype)
        if self.include_heterogeneous:
            self.y_info_mu = self.y_mu / self.y_var
        else:
            self.y_info_mu = self.y_mu
        if self.include_latent:
            self.num_latent = self.A.shape[1]
            assert init_len_in_mem >= self.num_latent
            self.info_mat.diagonal()[self.num_latent] = 1.0 / self.q_var
            self.q_info_mu = self.q_mu / self.q_var
        else:
            self.q_info_mu = self.q_mu
        
        self.update_method = update_method
        self.non_negative = non_negative
        self.num_replay_update = num_replay_update

    def ekf_step(self, obs_model: ObservationLinearization, measurement: torch.Tensor, verbose=False, replay=False):
        """
        Run a single EKF step that updates the full dense covariance matrix.
        """
        obs_idx = obs_model.state_indices
        obs_W_tsr = obs_model.matrix
        z = measurement.clone()
        if obs_model.bias is not None:
            z -= obs_model.bias
        Rinv = 1.0 / obs_model.var if obs_model.var.ndim == 1 else torch.linalg.inv(obs_model.var)
        Rinvz = z * Rinv if obs_model.var.ndim == 1 else Rinv @ z 

        y_mu_inc = (obs_W_tsr.T @ Rinvz)
        if self.include_heterogeneous:
            self.y_info_mu[obs_idx] += y_mu_inc
        if self.include_latent:
            q_mu_inc = (self.A[obs_idx, :].T @ obs_W_tsr @ Rinvz)
            self.q_info_mu += q_mu_inc

        if not replay:
            Wt_Rinv_W = (obs_W_tsr.T @ (Rinv[:,None] * obs_W_tsr)) if obs_model.var.ndim == 1 else (obs_W_tsr.T @ Rinv) @ obs_W_tsr
            touch_idx = self.get_touched_idx(obs_idx) + self.num_latent
            if self.include_latent:
                self.info_mat[:self.num_latent,:self.num_latent] += (self.A[obs_idx, :].T @ Wt_Rinv_W @ self.A[obs_idx, :])
            if self.include_heterogeneous:
                mat_idx = torch.meshgrid(touch_idx, touch_idx, indexing='ij')
                self.info_mat[mat_idx] += Wt_Rinv_W
            if self.include_latent and self.include_heterogeneous:
                mat_idx = torch.meshgrid(torch.arange(self.num_latent), touch_idx, indexing='ij')
                self.info_mat[mat_idx] += self.A[obs_idx,:].T @ Wt_Rinv_W
                mat_idx = torch.meshgrid(touch_idx, torch.arange(self.num_latent), indexing='ij')
                self.info_mat[mat_idx] += Wt_Rinv_W @ self.A[obs_idx,:]

    def update_estimation(self, verbose=False):
        self.ekf_step(*self.obs_buffer[-1], verbose)
            
        if self.num_replay_update > 0:
            step_idx_ary = np.random.randint(0, self.num_obs(), 
                                             (self.num_replay_update,))
            for step_idx in step_idx_ary:
                self.ekf_step(*self.obs_buffer[step_idx], verbose, replay=True)

    def finalize_estimation(self, verbose=False):        
        """
        Dense EKF finalization step requires solving a linear system.
        
        For computational efficiency, the covariance matrix are not saved
        during the online estimation stage. During finalization, we need 
        to compute the covariance matrix from the information matrix by solving
        a linear system.
        """
        # index conversion
        sorted_raw_idx = self.get_sorted_raw_idx()
        touch_info_vec = torch.concat([self.q_info_mu,self.y_info_mu[sorted_raw_idx]])

        num = self.num_latent + self.num_touched()
        touch_info_mat = self.info_mat[:num, :num]

        if self.update_method == 'gs':
            mu_hat = torch.zeros_like(touch_info_vec)
            for _ in range(self.iter_per_update):
                U = torch.triu(touch_info_mat, diagonal=1)
                res_vec = -U @ mu_hat + touch_info_vec
                torch.linalg.solve_triangular(torch.tril(touch_info_mat), res_vec, 
                                              upper=False, out=mu_hat)
        elif self.update_method == 'inv':
            mu_hat = torch.zeros_like(touch_info_vec)
            torch.linalg.solve(touch_info_mat, touch_info_vec, out=mu_hat)
        else:
            raise ValueError("Invalid update_method, must be gs or inv")
        
        if self.non_negative:
            mu_hat.clamp_(min=0)
        self.q_mu = mu_hat[:self.num_latent]
        self.y_mu[sorted_raw_idx] = mu_hat.reshape(-1)[self.num_latent:]

    def num_touched(self):
        """Return number of points in contact from the observation buffer"""
        return torch.sum(self.idx_raw2obs != -1).item()

    def get_touch_info_mat(self):
        """
        Return the information matrix for the touched indices
        
        NOTE: the information matrix is not the full dense matrix, we only
        need to index the left-top corner of the matrix with size num_touched x num_touched.
        """
        num = self.num_touched()
        return self.info_mat[:num, :num]
    
    def get_sorted_raw_idx(self):
        """Return the sorted raw index of the touched indices"""
        all_raw_idx   = torch.where(self.idx_raw2obs != -1)[0]
        idx_touch2raw = torch.argsort(self.idx_raw2obs[all_raw_idx])
        return all_raw_idx[idx_touch2raw]

    def get_touched_idx(self, obs_idx:torch.Tensor):
        """
        Return the touched index for the observation index
        
        NOTE: if the observation index is not in the touched index,
        this function will add the index to the touched index.
        
        TODO: finish the increment of the information matrix.
        """
        touched_idx = self.idx_raw2obs[obs_idx]
        none_idx = obs_idx[touched_idx == -1]

        if none_idx.size() != 0:
            add_idx = torch.arange(none_idx.shape[0], device=self.y_mu.device, dtype=int)
            add_idx += self.num_touched() 
            self.idx_raw2obs[none_idx] = add_idx

            if self.num_touched() > self.info_mat.shape[0]:
                #extend the information matrix with the new indices
                raise NotImplementedError("Dense EKF extension implemented yet")
                old_len = len(self.info_mat)
                new_len = old_len+self.mem_inc
                prior_info_val = self.prior_lam*self.prior_mu
                new_info_mat = self.prior_lam*torch.eye(new_len, **self.tsr_params)
                new_info_mat[:old_len, :old_len] = self.info_mat

                # update info vec and mat
                self.info_mat = new_info_mat
            
            # re-read touched index
            touched_idx = self.idx_raw2obs[obs_idx]

        return touched_idx

    def state_dict(self):
        """Get the state dictionary for the dense EKF estimator"""
        res = super(self).state_dict()
        res['info_mat'] = self.info_mat
        res['y_info_mu'] = self.y_info_mu
        res['q_info_mu'] = self.q_info_mu
        return res
    
    def load_state_dict(self, state_dict : dict):
        """Load the state dictionary for the dense EKF estimator"""
        super(self).load_state_dict(state_dict)
        self.info_mat = state_dict['info_mat']
        self.y_info_mu = state_dict['y_info_mu']
        self.q_info_mu = state_dict['q_info_mu']
