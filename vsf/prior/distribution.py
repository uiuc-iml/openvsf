from __future__ import annotations
from abc import ABC,abstractmethod
import torch
from typing import List, Callable

class ParamDistribution(ABC):
    """Base class of a distribution over a vector space.
    """

    def log_prob(self, param:torch.Tensor) -> torch.Tensor:
        """Returns the log probability of the given material parameters.
        
        Note: this function is supposed to be differentiable.
        """
        raise NotImplementedError()
    
    def param_mean(self) -> torch.Tensor:
        """Returns the mean estimation of the model parameters."""
        raise NotImplementedError()

    def param_std(self) -> torch.Tensor:
        """Returns the standard deviation estimation of the model parameters."""
        raise NotImplementedError()
    
    def __getitem__(self, idx_or_idxs) -> ParamDistribution:
        """Returns the distribution of the given indices.
        """
        raise NotImplementedError()


class DiagGaussianDistribution(ParamDistribution):
    """Gaussian distribution with diagonal covariance.
    """
    def __init__(self, mu:torch.Tensor, var:torch.Tensor) -> None:
        self.mu = mu
        assert len(var.shape) == 1
        self.var = var

    def log_prob(self, param: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.mu, self.var).log_prob(param)

    def param_mean(self):
        return self.mu
    
    def param_std(self):
        return torch.sqrt(self.var)

    def __getitem__(self, idx_or_idxs) -> DiagGaussianDistribution:
        return DiagGaussianDistribution(self.mu[idx_or_idxs], self.var[idx_or_idxs])

    def get_inv_var(self, eps:float=1e-6) -> torch.Tensor:
        """Returns the inverse of the covariance matrix."""
        return 1.0/ (self.var + eps)
    
    def tofull(self):
        """Converts this distribution to a full Gaussian distribution."""
        return FullGaussianDistribution(self.mu, torch.diag(self.var))
    
    def todiag(self):
        """Converts this distribution to a diagonal Gaussian distribution."""
        return self


class FullGaussianDistribution(ParamDistribution):
    """Gaussian distribution with non-diagonal covariance.
    """
    def __init__(self, mu:torch.Tensor, var:torch.Tensor) -> None:
        self.mu = mu
        assert var.shape == (len(mu),len(mu))
        self.var = var
        self.normal = None

    def log_prob(self, param: torch.Tensor) -> torch.Tensor:
        if self.normal is None:
            self.normal = torch.distributions.MultivariateNormal(self.mu, self.var)
        return self.normal.log_prob(param)

    def param_mean(self):
        return self.mu
    def param_std(self):
        return torch.sqrt(self.var.diag())
    
    def __getitem__(self, idx_or_idxs) -> FullGaussianDistribution:
        return FullGaussianDistribution(self.mu[idx_or_idxs], self.var[idx_or_idxs][:,idx_or_idxs])

    def get_inv_var(self, eps:float=1e-6) -> torch.Tensor:
        """Returns the inverse of the covariance matrix."""
        eps_mat = eps*torch.eye(self.var.shape[0], 
                                dtype=self.var.dtype, 
                                device=self.var.device)
        return torch.linalg.inv(self.var + eps_mat)

    def tofull(self):
        """Converts this distribution to a full Gaussian distribution."""
        return self
    
    def todiag(self):
        """Converts this distribution to a diagonal Gaussian distribution."""
        return DiagGaussianDistribution(self.mu, self.var.diag())