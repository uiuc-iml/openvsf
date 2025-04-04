import torch
from ..core.base_vsf import BaseVSF
from ..core.point_vsf import PointVSF
from .distribution import ParamDistribution,DiagGaussianDistribution
from .prior import LearnableContextualPrior, TorchLearnConfig
from abc import ABC,abstractmethod
from typing import List,Dict


class BaseVSFPriorFactory(ABC):
    """A factory class that creates a prior distribution for the VSF
    material parameters.
    """
    @abstractmethod
    def predict(self, vsf : BaseVSF,
                features : Dict[str,torch.Tensor]=None,
                material_param='stiffness') -> ParamDistribution:
        """Predicts the material parameters of the given VSF.
        
        If features are given, then the prior can use those features rather
        than the ones stored inside the `vsf.features` attribute.

        Future implementations may include a way to specify which material
        parameters to predict.  Current implementations just predict
        stiffness
        """
        raise NotImplementedError()
        
    @abstractmethod
    def load_state_dict(self,params_dict:dict) -> None:
        """Sets the parameters of the prior."""
        raise NotImplementedError()
    
    @abstractmethod
    def state_dict(self) -> dict:
        """Returns all the parameters of the prior as a dictionary."""
        raise NotImplementedError()


class GaussianVSFPriorFactory(BaseVSFPriorFactory):
    """A factory class that just predicts a Gaussian distribution for the
    material parameters.
    """
    def __init__(self, mean : float, var : float):
        self.mean = mean
        self.var = var
    
    def predict(self, vsf : BaseVSF, features : Dict[str,torch.Tensor]=None, material_param='stiffness') -> ParamDistribution:
        """Predicts the material parameters of the given VSF."""
        assert isinstance(vsf, PointVSF)
        return DiagGaussianDistribution(torch.full((len(vsf.rest_points),),self.mean,device=vsf.rest_points.device),
                                        torch.full((len(vsf.rest_points),),self.var,device=vsf.rest_points.device))
    
    def load_state_dict(self, params_dict):
        self.mean = params_dict['mean']
        self.var = params_dict['var']
    
    def state_dict(self) -> dict:
        return {'mean':self.mean, 'var':self.var}


class LearnableVSFPriorFactory(BaseVSFPriorFactory):
    """A base class for a learnable `VSFPriorFactory`.

    The predict_torch function must be differentiable through torch.
    """
    def __init__(self, feature_keys:List[str], prior : LearnableContextualPrior):
        self.feature_keys:List[str] = feature_keys
        self.prior:LearnableContextualPrior = prior
        self.learn_config = TorchLearnConfig()

    def predict(self, vsf:BaseVSF, features:Dict[str,torch.Tensor]=None, material_param='stiffness') -> ParamDistribution:
        """Predicts the material parameters of the given VSF."""
        context = self.feature_tensor(vsf,features,self.feature_keys)
        return self.prior.predict(context)

    def meta_learn(self, vsfs : List[BaseVSF],
              features : List[Dict[str,torch.Tensor]]=None,
              material_param='stiffness',
              weight_feature='N_obs'):
        """Meta-learns the prior from previously trained VSFs.
        
        Default incrementally learns from each of the VSFs in the
        list.
        """
        for vsf in vsfs:
            assert isinstance(vsf, PointVSF),"Only PointVSF is supported for now"
        context_tensors = [self.feature_tensor(vsf,features[i]) for i,vsf in enumerate(vsfs)]
        weight_tensors = [vsf.features[weight_feature] for vsf in vsfs]
        targets = [vsf.stiffness for vsf in vsfs]
        self.prior.learn(context_tensors, targets, weights=weight_tensors, learn_config=self.learn_config)

    def meta_learn_increment(self, vsf : BaseVSF, features : Dict[str,torch.Tensor] = None, material_params='stiffness'):
        """Incrementally learns the prior from a single VSF."""
        assert isinstance(vsf, PointVSF),"Only PointVSF is supported for now"
        context = self.feature_tensor(vsf,features,self.feature_keys)
        weight = vsf.features['N_obs'] if 'N_obs' in vsf.features else 1.0
        self.prior.learn_increment(context, vsf.stiffness, weight)

    def parameters(self) -> list:
        """Returns the torch parameters that can be optimized, i.e., same as
        torch.nn.Module.parameters()."""
        return self.prior.parameters()

    def to(self, device):
        """Moves the meta prior to the given device / dtype and return self."""
        self.prior.to(device)
        return self
    
    def eval(self):
        """Sets the model to evaluation mode."""
        self.prior.eval()

    def train(self):
        """Enables torch autograd with respect to the model parameters."""
        self.prior.train()

    def feature_tensor(self, vsf : BaseVSF, features : Dict[str,torch.Tensor]) -> torch.Tensor:
        """Helper: returns the feature tensor corresponding to the given keys.
        
        Result is an Nxf tensor, where N is the number of points and f is the
        number of features.
        """
        assert isinstance(vsf, PointVSF)
        assert len(self.feature_keys) > 0
        selected = [features[key] if key in features else vsf.features[key] for key in self.feature_keys]
        selected_reshaped = []  #make 2D tensors
        for f in selected:
            assert f.shape[0] == selected.shape[0]
            selected_reshaped.append(f.reshape(selected.shape[0],-1))
        return torch.cat(selected_reshaped,dim=1)

    def feature_tensor_batch(self, vsfs : List[BaseVSF], features : List[Dict[str,torch.Tensor]]) -> torch.Tensor:
        """Helper: returns the feature tensor corresponding to the given keys
        for a batch of VSFs.
        
        Result is a BxNxf tensor, where N is the number of points and f is the
        number of features.
        """
        selected = [self.feature_tensor(vsf,features[i]) for i,vsf in enumerate(vsfs)]
        return torch.stack(selected)

