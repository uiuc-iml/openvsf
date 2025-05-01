import torch
from torch import nn
from .distribution import ParamDistribution, DiagGaussianDistribution, FullGaussianDistribution
from dataclasses import dataclass
from abc import ABC,abstractmethod
from typing import Dict, List, Union


class BaseConditionalDistribution(ABC):
    """
    A base class for a conditional distribution that predicts the distribution 
    of a generic vector given a feature tensor.
    
    The contextual prior works on fixed-dimensional input features and 
    output material parameters.  
    
    Inputs to the conditional distribution can be in a batch form,
    i.e., a B x F tensor, where B is the batch size and F is the dimension 
    of the features.  The output of the contextual prior is a distribution
    over the material parameters.

    """
    @abstractmethod
    def predict(self, context:torch.Tensor) -> ParamDistribution:
        """Predicts the output for a properly-shaped context tensor.
        
        Args:
            context: A B x F tensor, where B is the batch size and F is the
                number of features.
        
        Returns:
            A ParamDistribution object representing the predicted distribution
            over B outputs.
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


class LearnableConditionalDistribution(BaseConditionalDistribution):
    """A conditional distribution that can be learned from data.

    It is assumed that the predict function is differentiable through torch.
    """

    def learn(self, contexts : List[torch.Tensor], targets : List[torch.Tensor], weights : List[torch.Tensor]=None):
        """Learns the prior from context / target pairs.  Weights are optional.

        Default implementation is to call learn_increment for each pair.
        """
        weights = weights if weights is not None else [None]*len(contexts)
        for context,target,weight in zip(contexts, targets, weights):
            self.learn_increment(context,target,weight)

    @abstractmethod
    def learn_increment(self, context : torch.Tensor, target : torch.Tensor, weight : torch.Tensor = None):
        """Incrementally learns the prior from a single context / target pair."""
        raise NotImplementedError()

    @abstractmethod
    def parameters(self) -> list:
        """Returns the torch parameters that can be optimized, i.e., same as
        torch.nn.Module.parameters().
        
        NOTE: this function is expected to return references of the parameters,
        different from state_dict() which returns copies of the parameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def to(self, device):
        """Moves the meta prior to the given device / dtype and return self."""
        raise NotImplementedError()
    
    @abstractmethod
    def eval(self):
        """Sets the model to evaluation mode."""
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """Enables torch autograd with respect to the model parameters."""
        raise NotImplementedError()


class GaussianConditionalDistribution(LearnableConditionalDistribution):
    """A contextual prior that predicts the mean and variance of the output.
    
    The mean and variance are learned through simple estimation.  This also
    supports PyTorch gradient descent learning.
    """
    def __init__(self, mean_init = 0.0, var_init = 1.0):
        self.mean = nn.Parameter(torch.tensor(mean_init))
        self.var = nn.Parameter(torch.tensor(var_init))

    def learn(self, contexts : List[torch.Tensor], targets : List[torch.Tensor], weights : List[torch.Tensor]=None):
        self.mean.data = torch.mean(torch.cat(targets))
        self.var.data = torch.var(torch.cat(targets))
    
    def parameters(self):
        return [self.mean, self.var]
    
    def learn_increment(self, context : torch.Tensor, target : torch.Tensor, weight : torch.Tensor = None):
        raise NotImplementedError("GaussianConditionalDistribution does not support incremental learning")

    def predict(self, context:torch.Tensor) -> ParamDistribution:
        ones = torch.ones(context.size(0), device=context.device)
        return DiagGaussianDistribution(self.mean*ones, self.var*ones)
    
    def to(self, device):
        self.mean.to(device)
        self.var.to(device)
        return self
    
    def eval(self):
        self.mean.requires_grad = False
        self.var.requires_grad = False
    
    def train(self):
        self.mean.requires_grad = True
        self.var.requires_grad = True
    
    def load_state_dict(self, params_dict:dict):
        self.mean = params_dict.get('mean')
        self.var = params_dict.get('var')

    def state_dict(self):
        return {
            'mean': self.mean, 'var': self.var
        }

@dataclass
class TorchLearnConfig:
    """A configuration for standard torch stochastic gradient descent
    optimization witth batches."""
    batch_size : int = 32
    shuffle : bool = True
    optimizer : str = 'adam'
    weight_decay : float = 0.0
    momentum : float = 0.9
    num_epochs : int = 100
    lr : float = 1e-3
    lr_decay_factor : float = 0.1
    lr_decay_patience : int = 0
    lr_decay_min : float = 1e-6


class TorchConditionalDistribution(LearnableConditionalDistribution):
    """A conditional distribution that uses two torch modules to predict the mean and
    variance of the material parameters.

    Args:
        mean_module: A torch module that predicts the mean of the material
            parameters.  It should take a B x N x F tensor as input, where
            B is the batch size, N is the number of points, and F is the
            number of features.
        var_module: A torch tensor or module that predicts the variance of
            the material parameters. If a tensor, it is used as a constant
            variance.  
        var_diag: If True, the variance module should give the diagonal
            of the covariance matrix.  Otherwise, it should give the full
            covariance matrix.

    """
    def __init__(self, mean_module:nn.Module, var_module:Union[nn.Module,torch.Tensor], var_diag=False):
        self.mean_module:nn.Module = mean_module
        self.var_module:nn.Module = var_module
        self.var_diag:bool = var_diag
        self.num_obs:int = 0

    def learn(self, contexts : List[torch.Tensor], targets : List[torch.Tensor], weights : List[torch.Tensor]=None,
              learn_config=TorchLearnConfig()):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        
        dataset = TensorDataset(torch.cat(contexts), torch.cat(targets), torch.cat(weights))
        dataloader = DataLoader(dataset, batch_size=learn_config.batch_size, shuffle=learn_config.shuffle)
        if learn_config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learn_config.lr, weight_decay=learn_config.weight_decay)
        elif learn_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learn_config.lr, momentum=learn_config.momentum, weight_decay=learn_config.weight_decay)
        else:
            raise ValueError("Invalid optimizer: "+learn_config.optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learn_config.lr_decay_factor, patience=learn_config.lr_decay_patience, min_lr=learn_config.lr_decay_min)
        if callable(self.var_module):
            criterion = nn.GaussianNLLLoss()
        else:
            criterion = nn.MSELoss()
        self.train()
        for epoch in range(learn_config.num_epochs):
            for context_batch,target_batch,weight_batch in dataloader:
                optimizer.zero_grad()
                prior_mu = self.mean_module(context_batch)
                if callable(self.var_module):
                    prior_var = self.var_module(context_batch)
                    loss = criterion(prior_mu, target_batch, prior_var)
                else:
                    loss = criterion(prior_mu, target_batch)
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        self.eval()
        if not callable(self.var_module):
            #fit var_module to the residuals
            #TODO: weights
            residuals = []
            for ctx,tgt in zip(contexts,targets):
                prior_mu = self.mean_module(ctx)
                residuals.append(tgt - prior_mu)
            residuals = torch.cat(residuals)
            if self.var_diag:
                new_var = torch.sum(residuals**2,dim=0)/(len(residuals)-1)
            else:
                new_var = torch.sum([r @ r.T for r in residuals],dim=0)/(len(residuals)-1)
            #use num_obs to do a running average
            if self.num_obs == 0:
                self.var_module = new_var
            else:
                #note: this does not properly handle the mean changing
                self.var_module = (self.var_module*self.num_obs + new_var*len(residuals))/(self.num_obs+len(residuals))
            self.num_obs += len(contexts)
        return

    def learn_increment(self, context: torch.Tensor, target: torch.Tensor, weight=1.0, learn_config=TorchLearnConfig(num_epochs=1)):
        config = TorchLearnConfig(**learn_config)
        config.batch_size = 1
        config.shuffle = False
        self.learn([context], [target], [weight], learn_config=config)

    def to(self, device):
        self.mean_module.to(device)
        self.var_module.to(device)
        return self

    def eval(self):
        self.mean_module.eval()
        self.var_module.eval()

    def train(self):
        self.mean_module.train()
        self.var_module.train()
    
    def state_dict(self):
        return {'mean_module': self.mean_module.state_dict(),
                'var_module': self.var_module.state_dict()}

    def load_state_dict(self,params_dict):
        self.mean_module.load_state_dict(params_dict['mean_module'])
        self.var_module.load_state_dict(params_dict['var_module'])
    
    def predict(self, context:torch.Tensor) -> ParamDistribution:
        """Predicts the material parameters of the given features."""
        return self.predict_torch(context)

    def predict_torch(self, context:torch.Tensor) -> ParamDistribution:
        """Predicts the material parameters of the given features."""
        prior_mu = self.mean_module(context)
        prior_var = self.var_module(context)
        if self.var_diag:
            return DiagGaussianDistribution(prior_mu,prior_var)
        else:
            return FullGaussianDistribution(prior_mu,prior_var)

    def predict_mean(self,context:torch.Tensor) -> torch.Tensor:
        """Predicts the material parameters of the given dataset."""
        return self.mean_module(context)
    
    def predict_var(self,context:torch.Tensor) -> torch.Tensor:
        """Predicts the variance of material parameters for the given dataset."""
        return self.var_module(context)
            
    def parameters(self):
        return list(self.mean_module.parameters()) + list(self.var_module.parameters())
    


@dataclass
class LinearGaussianPriorConfig:
    """A configuration for a linear Gaussian conditional distribution."""
    c_dim:int = None
    diag:bool = False
    non_neg:bool = False
    initial_weight_scale:float = 1e-4
    mu_init:float = 1e-2
    var_init:float = 1.0

class ConstantOutput(nn.Module):
    """
    A module that outputs constant values for all inputs.
    
    This module is a temprary solution to be compatible with the
    var_module in TorchConditionalDistribution.
    """
    def __init__(self, constant_value: torch.Tensor):
        super().__init__()
        self.constant = nn.Parameter(constant_value, requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns the constant value for all inputs."""
        return self.constant.expand(features.size(0), -1).flatten()

class LinearGaussianPrior(TorchConditionalDistribution):
    """A meta-prior that predicts the material parameters as a linear
    function of the named features plus a constant gaussian uncertainty.
    """
    def __init__(self, config : LinearGaussianPriorConfig) -> None:
        self.config = config

        linear = nn.Linear(config.c_dim, 1)
        torch.nn.init.xavier_uniform_(linear.weight.data, config.initial_weight_scale)
        torch.nn.init.constant_(linear.bias.data, config.mu_init)

        if config.non_neg:
            non_neg = nn.LeakyReLU(negative_slope=-0.5)
            mean_module = nn.Sequential(linear,non_neg,nn.Flatten(start_dim=0))
        else:
            mean_module = linear
        
        # nn.Module cannot be a torch tensor
        # if config.diag:
        #     var_module = torch.full((config.c_dim,),config.var_init)
        # else:
        #     var_module = torch.eye(config.c_dim)*config.var_init
        var_module = ConstantOutput(torch.Tensor([config.var_init]))
        super().__init__(mean_module, var_module, config.diag)

    
    # Shaoxiong's earlier stuff.  There's some RBF kernel stuff in here that I'm not sure is necessary.
    # def predict_sig(self,context:torch.Tensor) -> torch.Tensor:
    #     """Predicts the material parameters of the given dataset."""

    #     num_pts = context.shape[0]
    #     if self.diag:
    #         prior_sig = self.meta_sig*torch.ones(num_pts,)
    #     else:
    #         prior_sig = self.meta_sig*torch.eye(num_pts)
    #         if self.kernel_params != {}:
    #             if self.kernel_params['type'] == 'rbf':
    #                 kernel_scale = self.kernel_params['scale']
    #                 sig_scale = self.kernel_params['sig_scale']
    #                 dist_mat = torch.cdist(context, context, p=2, 
    #                                        compute_mode='donot_use_mm_for_euclid_dist')
    #                 prior_sig += sig_scale*torch.exp(-dist_mat**2 / kernel_scale**2)
    #     if self.exp_sig:
    #         return torch.exp(prior_sig)
    #     else:
    #         return prior_sig
        
    def get_default_opt(self) -> List[dict]:
        return [
            {'params': [self.c2m_mat], 'lr': 1e-4},
            {'params': [self.meta_sig], 'lr': 0.001}
        ]


class MLP(nn.Module):
    def __init__(self, dim_lst, non_neg=False, add_bn=True, 
                 add_dropout=True, add_res=False, drop_p:float=0.5, 
                 init_params:dict={}):
        super(MLP, self).__init__()

        self.layers_lst = [nn.Linear(dim_lst[0], dim_lst[1])]
        # if add_dropout:
        #     layers_lst.append(nn.Dropout(p=0.5))
        for dim1, dim2 in zip(dim_lst[1:-1], dim_lst[2:]):
            if add_bn:
                self.layers_lst.append(nn.BatchNorm1d(dim1))
            self.layers_lst.append(nn.ReLU())
            self.layers_lst.append(nn.Linear(dim1, dim2))
            if add_dropout:
                self.layers_lst.append(nn.Dropout(p=drop_p))
        self.layers = nn.Sequential(*self.layers_lst)

        self.add_res = add_res
        if self.add_res:
            self.res = nn.Linear(dim_lst[0], dim_lst[-1])

        self.non_neg = non_neg
        if non_neg:
            self.non_neg_layer = nn.LeakyReLU(negative_slope=-0.01)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.add_res:
            nn.init.xavier_uniform_(self.res.weight)
            nn.init.zeros_(self.res.bias)
        if add_bn and 'out_scale' in init_params:
            self.layers_lst[-1].weight.data = torch.tensor(init_params['out_scale'])
        if 'out_bias' in init_params:
            last_id = -2 if add_dropout else -1
            self.layers_lst[last_id].bias.data = torch.tensor(init_params['out_bias'])

    def forward(self, x):
        out = self.layers(x)
        if self.add_res:
            out += self.res(x)
        if self.non_neg:
            out = self.non_neg_layer(out)
        return out

class MlpConditionalDistribution(TorchConditionalDistribution):

    def __init__(self, c_dim:int=None, dim_lst:list[int]=None, 
                 tsr_params:dict={}, non_neg:bool=False, add_bn:bool=True, 
                 add_res:bool=False, add_dropout:bool=True, init_params:dict={}) -> None:
        self.c_dim:int = c_dim
        self.tsr_params:dict = tsr_params
        self.non_neg:bool = non_neg

        assert dim_lst[-1] == 2

        self.mlp = MLP(dim_lst, non_neg=non_neg, add_bn=add_bn, 
                       add_res=add_res, add_dropout=add_dropout, 
                       init_params=init_params)
        self.mlp.to(**tsr_params)

    def predict(self,context:torch.Tensor) -> DiagGaussianDistribution:
        """Predicts the material parameters of the given dataset.
           context dimension: (num_pts, c_dim)
           output: GaussianDistribution(mu, sig)
        """
        mlp_out = self.mlp(context)
        prior_mu, prior_sig = mlp_out[:, 0], mlp_out[:, 1]
        return DiagGaussianDistribution(prior_mu, prior_sig)
    
    def predict_mean(self, context:torch.Tensor) -> torch.Tensor:
        """Predicts the material parameters of the given dataset."""
        mlp_out = self.mlp(context)
        return mlp_out[:, 0]
    
    def eval(self):
        self.mlp.eval()
    
    def train(self):
        self.mlp.train()

    def parameters(self):
        return self.mlp.parameters()

    def state_dict(self, data_type) -> dict:
        return self.mlp.state_dict()
    
    def load_state_dict(self, params_dict:dict) -> None:
        self.mlp.load_state_dict(params_dict)


class DeepKernelConditionalDistribution(TorchConditionalDistribution):

    def __init__(self, c_dim:int=None, dim_lst:list[int]=None, 
                 tsr_params:dict={}, non_neg:bool=False) -> None:
        self.c_dim:int = c_dim
        self.tsr_params:dict = tsr_params
        self.non_neg:bool = non_neg

        assert dim_lst[-1] >= 2

        self.non_neg = non_neg
        self.mlp = MLP(dim_lst, non_neg=False)
        self.mlp.to(**tsr_params)

        self.sig_scale = torch.tensor(1.0, **tsr_params)

    def predict(self,context:torch.Tensor) -> ParamDistribution:
        """Predicts the material parameters of the given dataset.
           context dimension: (num_pts, c_dim)
           output: GaussianDistribution(mu, sig)
        """
        mlp_out = self.mlp(context)
        prior_mu, prior_feats = mlp_out[:, 0], mlp_out[:, 1:]

        if self.non_neg:
            prior_mu = torch.nn.LeakyReLU(negative_slope=-0.05)(prior_mu)
        # NOTE: not using mm is critical for float type data
        prior_sig = torch.cdist(prior_feats, prior_feats, p=2,
                                compute_mode='donot_use_mm_for_euclid_dist')
        # if not torch.allclose(prior_sig, prior_sig.T):
        #     np.save('prior_feats.npy', prior_feats.detach().cpu().numpy())
        assert torch.allclose(prior_sig, prior_sig.T)
        
        prior_sig = self.sig_scale*torch.exp(-prior_sig**2)
        assert torch.allclose(prior_sig, prior_sig.T)

        return FullGaussianDistribution(prior_mu, prior_sig)

    def predict_mean(self, context:torch.Tensor) -> torch.Tensor:
        """Predicts the material parameters of the given dataset."""
        mlp_out = self.mlp(context)
        if self.non_neg:
            prior_mu = torch.nn.LeakyReLU(negative_slope=-0.05)(mlp_out[:, 0])
        return prior_mu
    
    def predict_corr_sig(self, context1:torch.Tensor, context2:torch.Tensor) -> torch.Tensor:
        """Predicts the correlation matrix between the given contexts."""
        mlp_out1 = self.mlp(context1)
        prior_feats1 = mlp_out1[:, 1:]

        mlp_out2 = self.mlp(context2)
        prior_feats2 = mlp_out2[:, 1:]

        dist_mat = torch.cdist(prior_feats1, prior_feats2, p=2,
                               compute_mode='donot_use_mm_for_euclid_dist')
        corr_sig = self.sig_scale*torch.exp(-dist_mat**2)
        return corr_sig

    def eval(self):
        self.mlp.eval()
        self.sig_scale.requires_grad = False
    
    def train(self):
        self.mlp.train()
        self.sig_scale.requires_grad = True

    def parameters(self):
        return list(self.mlp.parameters()) + [self.sig_scale]
    
    def state_dict(self, data_type) -> dict:
        state_dict = self.mlp.state_dict()
        state_dict['sig_scale'] = self.sig_scale
        return state_dict
    
    def load_state_dict(self, params_dict:dict) -> None:
        self.sig_scale = params_dict['sig_scale']
        del params_dict['sig_scale']
        self.mlp.load_state_dict(params_dict)
    

