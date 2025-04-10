from __future__ import annotations
from .base_vsf import BaseVSF
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field

@dataclass
class NeuralVSFConfig:
    """Configuration for a neural VSF model."""
    aabb: Tuple[List[float], List[float]] = field(default_factory=lambda: ([-1., -1., -1.], [1., 1., 1.]))
    """The domain of the vsf.  Output values are 0 outside of this domain"""
    output_names: List[str] = field(default_factory=lambda: ['stiffness'])
    """The names of the output fields.  Each field can have multiple dimensions as given by the output_dims attribute."""
    output_dims: List[int] = field(default_factory=lambda:[1])
    """The dimensions of each output field.  The length of this list should match the length of the output_names attribute."""
    num_layers: int = 8
    """The number of layers in the neural network."""
    hidden_dim: int = 64
    """The hidden dimension of the neural network."""
    skip_connection: List[int] = field(default_factory=lambda: [4])
    """The layers to add skip connections to."""
    output_scale: float = 1e4
    """Multiply the network output by a constant factor."""


# borrowed (and modified) from https://github.com/ashawkey/torch-ngp
class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    # hashgrid encoding requires the gridencoder module
    # elif encoding == 'hashgrid':
    #     from gridencoder import GridEncoder
    #     encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim


class VSFNetwork(nn.Module):
    def __init__(self,
                 num_layers=8,
                 hidden_dim=64,
                 skip_connection=[4],
                 output_names=['stiffness'],
                 output_dims=[1],
                 aabb=None,
                 sdf=None,
                 output_scale=1e4,
                 **kwargs
                 ):
        
        super(VSFNetwork, self).__init__()

        if aabb is None:
            print("Warning: AABB not provided, using default AABB")
            aabb = [[-1., -1., -1.], [1., 1., 1.]]
        aabb = torch.tensor(aabb, dtype=torch.float32)
        center = .5 * (aabb[0] + aabb[1])
        scale = (aabb[1] - aabb[0]) / 2.
        self.aabb = aabb
        self.center = center
        self.scale = scale
        self.sdf = sdf

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connection = skip_connection
        self.output_names = output_names
        self.output_dims = output_dims
        self.output_dim = sum(output_dims)
        self.encoder, self.in_dim = get_encoder('frequency', input_dim=3, multires=4)
        # self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, num_levels=8, level_dim=2, base_resolution=16, log2_hashmap_size=12, desired_resolution=512)
        self.output_scale = output_scale

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.output_dim
            else:
                out_dim = hidden_dim
            
            if l in skip_connection:
                sigma_net.append(nn.Linear(in_dim + self.in_dim, out_dim, bias=False))
            else:
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)


    def forward(self, x):
        x = (x - self.center) / self.scale
        bound_mask = (x[..., 0] > -1) & (x[..., 1] > -1) & (x[..., 2] > -1) & \
                     (x[..., 0] <  1) & (x[..., 1] <  1) & (x[..., 2] <  1)
        
        if self.sdf is not None:
            x_shape = x.shape
            x_sdf = torch.nn.functional.grid_sample(self.sdf[None, None, ...],
                                                    x.flip(dims=(-1,)).reshape(1,-1,1,1,3), mode='bilinear', align_corners=True).reshape(x_shape[:-1])
            sdf_mask = x_sdf < 0
            bound_mask = bound_mask & sdf_mask

        sigma_r = torch.zeros(*x.shape[:-1], 1, device=x.device)
        x = x[bound_mask]

        # sigma
        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skip_connection:
                h = torch.cat([h, x], dim=-1)
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = torch.exp(h)
        sigma_r[bound_mask] = sigma

        return sigma_r * self.output_scale
    
    def get_sdf(self, x):
        assert self.sdf is not None, "SDF not provided"
        x = (x - self.center) / self.scale
        
        x_shape = x.shape
        x_sdf = torch.nn.functional.grid_sample(self.sdf[None, None, ...],
                                                x.flip(dims=(-1,)).reshape(1,-1,1,1,3), mode='bilinear', align_corners=True).reshape(x_shape[:-1])
        return x_sdf


    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]
        
        return params
    
    def save(self, path):
        save_dict = {
            "save_dict": self.state_dict(),
            "aabb": self.aabb,
            "center": self.center,
            "scale": self.scale,
            "sdf": self.sdf,
            "output_scale": self.output_scale
        }
        torch.save(save_dict, path)

    def load(self, path):
        load_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(load_dict["save_dict"])
        self.aabb = load_dict["aabb"]
        self.center = load_dict["center"]
        self.scale = load_dict["scale"]
        self.sdf = load_dict["sdf"]
        self.output_scale = load_dict["output_scale"]
        self.eval()

    def to(self, device):
        self.aabb = self.aabb.to(device)
        self.center = self.center.to(device)
        self.scale = self.scale.to(device)
        if self.sdf is not None:
            self.sdf = self.sdf.to(device)
        return super().to(device)

    @property
    def device(self):
        return next(self.parameters()).device


class NeuralVSF(BaseVSF):
    """
    A Neural VSF model that conforms to the BaseSDF base class.

    Optionally takes an SDF tensor to use as a ground truth SDF for training.

    :param vsfConfig: Configuration for the Neural VSF model.
    :type vsfConfig: dict

    :param sdf: Optional SDF tensor to use as a geometry mask. 
                The SDF values are sampled at the center of the grid cells.
                Cell (0,0,0) is at `BBox[0]` and cell `(N-1, N-1, N-1)` is at `BBox[1]`.
    :type sdf: torch.Tensor, optional

    """
    def __init__(self, vsfConfig : NeuralVSFConfig, sdf : torch.Tensor = None):
        super().__init__()
        self.config = vsfConfig
        self.vsfNetwork = VSFNetwork(sdf=sdf, **asdict(vsfConfig))

    def getBBox(self) -> torch.Tensor:
        return torch.tensor(self.config.aabb)

    def getStiffness(self, position: torch.Tensor) -> torch.Tensor:
        assert self.config.output_names[0] == 'stiffness',"We only support stiffness as the first output"
        assert self.config.output_dims[0] == 1
        return self.vsfNetwork(position)[:,0]

    def save(self, path):
        self.vsfNetwork.save(path)

    def load(self, path):
        self.vsfNetwork.load(path)

    def to(self, device) -> NeuralVSF:
        """Converts the VSF to a given device or dtype"""
        self.vsfNetwork.to(device)
        return self

    @property
    def device(self):
        return next(self.vsfNetwork.parameters()).device
    
    @property
    def dtype(self):
        return next(self.vsfNetwork.parameters()).dtype