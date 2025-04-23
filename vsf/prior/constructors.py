from .prior_factory import BaseVSFPriorFactory, GaussianVSFPriorFactory, LearnableVSFPriorFactory
from .conditional_distribution import BaseConditionalDistribution, LearnableConditionalDistribution, GaussianConditionalDistribution
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class PriorConfig:
    """A configuration for a prior."""
    
    type : str
    """type of prior, e.g., 'gaussian', 'linear_gaussian'"""
    mu : float
    """initial mean"""
    var : float
    """initial variance"""
    diag : bool = True
    """whether the covariance is diagonal"""
    c_dim : int = 0
    """context dimension"""


@dataclass
class PointVSFPriorConfig(PriorConfig):
    """A configuration for a prior factory."""
    feature_keys : List[str] = field(default_factory=list)  # feature keys for the prior
    
def make_prior(config: PriorConfig) -> BaseConditionalDistribution:
    if config.type == 'gaussian':
        return GaussianConditionalDistribution(config.mu,config.var)
    elif config.type == 'linear_gaussian':
        from .conditional_distribution import LinearGaussianPrior,LinearGaussianPriorConfig
        return LinearGaussianPrior(LinearGaussianPriorConfig(c_dim=config.c_dim,diag=config.diag,mu_init=config.mu,var_init=config.var))
    raise ValueError("Unknown prior type "+config.type)

def make_prior_factory(config : PointVSFPriorConfig) -> BaseVSFPriorFactory:
    if config.type == 'gaussian':
        return GaussianVSFPriorFactory(config.mu,config.var)
    else:
        return LearnableVSFPriorFactory(config.feature_keys, make_prior(config))
