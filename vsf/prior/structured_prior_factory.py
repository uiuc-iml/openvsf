import torch
from .prior_factory import LearnableVSFPriorFactory
from .conditional_distribution import LearnableConditionalDistribution, BaseConditionalDistribution, GaussianConditionalDistribution
from .distribution import ParamDistribution, DiagGaussianDistribution
from ..core.base_vsf import BaseVSF
from ..core.point_vsf import PointVSF
from dataclasses import dataclass
from abc import ABC,abstractmethod
from typing import Dict, List, Tuple, Union

class BaseVSFStructuredPriorFactory(ABC):
    """A base class for a meta prior that determines a latent per-vsf
    material distribution.

    The idea is to define for each VSF a latent space phi such that the stiffness
    of the VSF is then K_points + A*phi.  Here A(vsf) is a matrix defined by
    a fixed function of the VSF's rest points and features, and phi can be variable
    dimension between VSFs.   K_points is a per-point stiffness with independent
    priors.

    To learn the meta prior, we need a map from a *universal* latent space psi 
    to the object-specific latent space phi.  The meta prior then learns a
    distribution over psi.

    The map from psi to phi is defined by the S matrix.  S(vsf) is a fixed function 
    of the VSF's rest points and features.  P(phi_i) = P(S_i^T psi | A_i^+ f(VSF)).
    where f(VSF) gives the selected features of the VSF.

    As an example, a homogeneous prior would have S = I[1x1] and A = 1[Nx1].  The psi
    space would be 1-dimensional, and the prior would be a distribution over a VSF's
    average stiffness.

    A more complex example could be a segmentation of the VSF into different regions,
    with phi being a length-s vector of stiffnesses for each region.  psi would be
    a 1-D variable, S would be 1[sx1], and A would be the (N x s) indicator variable
    for each region.
    """

    @abstractmethod
    def phi_prior(self, vsf : BaseVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> ParamDistribution:
        """Returns the prior for the latent space phi."""
        raise NotImplementedError()
    
    def phi_basis(self, vsf : BaseVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> torch.Tensor:
        """Returns the basis of the latent space phi for the given VSF, i.e. the matrix A."""
        return self.phi_psi_bases(vsf,features,material_param)[0]
    
    @abstractmethod
    def phi_psi_bases(self, vsf : BaseVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> Tuple[torch.Tensor,torch.Tensor]:
        """Returns the bases of the latent spaces phi and psi for the given,
        VSF, i.e. the matrices A and S."""
        raise NotImplementedError()


class HomogeneousVSFPriorFactory(BaseVSFStructuredPriorFactory):
    """A meta prior factory that assumes a homogeneous stiffness
    distribution over the VSF.
    
    The constructor can take in a mean and variance for the prior,
    or a LearnableConditionalDistribution.
    """
    def __init__(self, mean_or_gaussian_prior : Union[float,LearnableConditionalDistribution], var : float = 1.0, feature_keys:List[str]=None):
        if isinstance(mean_or_gaussian_prior,float):
            self.prior = LearnableVSFPriorFactory([],GaussianConditionalDistribution(mean_or_gaussian_prior, var))
        else:
            if feature_keys is None:
                feature_keys = []
            self.prior = LearnableVSFPriorFactory(feature_keys,mean_or_gaussian_prior)
    
    def phi_prior(self, vsf : PointVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> ParamDistribution:
        """Returns the prior for the latent space phi."""
        if len(self.prior.feature_keys) is None:
            #gaussian prior over single variable
            context = torch.ones(1,1,device=vsf.rest_points.device)
            return self.prior.prior.predict(context)
        else:
            context = self.prior.feature_tensor(vsf,features,self.prior.feature_keys)
            mean_context = torch.mean(context,dim=0,keepdim=True)
            return self.prior.prior.predict(mean_context)
    
    def phi_psi_bases(self, vsf : PointVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> Tuple[torch.Tensor,torch.Tensor]:
        """Returns the bases of the latent spaces phi and psi for the given,
        VSF, i.e. the matrices A and S."""
        return torch.ones(vsf.rest_points.size(0),1,device=vsf.rest_points.device),torch.ones(1,1,device=vsf.rest_points.device)

    def phi_basis(self, vsf : PointVSF, features = None, material_param='stiffness'):
        return torch.ones(vsf.rest_points.size(0),1,device=vsf.rest_points.device)


class SegmentationVSFPriorFactory(BaseVSFStructuredPriorFactory):
    """A meta prior factory that segments a VSF into different regions and
    learns a stiffness distribution for each region.
    
    If homogeneous_segment is True, then the prior includes a uniform
    distribution over the average stiffness of the VSF.

    The segment_feature is the key in the VSF's features that indicates
    the segmentation of the VSF.  If the feature is not present, an error
    is raised.  The value of the feature should be an integer that indicates
    the segment ID.

    prior_feature_keys is a list of keys in the VSF's features that are used
    to predict the stiffness distribution.  If None, then the prior is assumed
    not contextual.
    """
    def __init__(self, prior : LearnableConditionalDistribution,
                 homogeneous_segment : bool = True,
                 segment_feature  = 'segment',
                 prior_feature_keys : List[str] = None):
        if prior_feature_keys is None:
            prior_feature_keys = []
        self.prior = LearnableVSFPriorFactory(prior_feature_keys,prior)
        self.homogeneous_segment = homogeneous_segment
        self.segment_feature = segment_feature

    def phi_prior(self, vsf : PointVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> ParamDistribution:
        """Returns the prior for the latent space phi."""
        features = self.prior.feature_tensor(vsf,vsf.features,self.prior.feature_keys)
        A,S = self.phi_psi_bases(vsf,features,material_param)
        A_inv = torch.linalg.pinv(A)
        f_phi = torch.matmul(A_inv,features)
        return self.prior.prior.predict(f_phi)
    
    def phi_psi_bases(self, vsf : PointVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> Tuple[torch.Tensor,torch.Tensor]:
        A_components = []
        S_entries = []
        if self.homogeneous_segment:
            A_components.append(torch.ones(vsf.rest_points.size(0),1,device=vsf.rest_points.device))
            S_entries.append([0])

        segs = None
        if self.segment_feature in features:
            segs = features[self.segment_feature]
        elif self.segment_feature in vsf.features:
            segs = vsf.features[self.segment_feature]
        if segs is None and self.segment_feature is not None:
            raise ValueError(f"Segmentation feature {self.segment_feature} not found in features or vsf")
        if segs is not None:
            seg_ids = segs.unique()
            S_entries.append(list(range(seg_ids.size(0))))
            for seg in seg_ids:
                mask = (segs == seg).float()
                A_components.append(mask)
        S_components = torch.zeros((len(A_components),len(S_entries)),device=vsf.rest_points.device)
        for i,entries in enumerate(S_entries):
            S_components[entries,i] = 1
        return torch.cat(A_components,dim=1),S_components
