from __future__ import annotations
import torch
import os
from .base_vsf import BaseVSF
import numpy as np
from typing import Union, Optional, Dict, Tuple, List
from dataclasses import dataclass, asdict, field


@dataclass
class PointVSFConfig:
    """Configuration for a point-based VSF model.

    Can pass these to PointVSF constructor via
    `PointVSF(**dataclasses.asdict(config))`.
    """
    bbox: Optional[np.ndarray] = None
    """bounding box of the VSF"""
    voxel_size: Optional[float] = None
    """voxel size of the VSF"""
    rest_points: Optional[Union[np.ndarray, torch.Tensor]
                          ] = None 
    """rest point locations"""
    axis_mode: str = 'isotropic'
    """how the stiffness is defined; can be isotropic, axis_aligned, or anisotropic"""
    features: Dict[str, Union[np.ndarray, torch.Tensor]] = field(
        default_factory=dict) 
    """optional per-point features"""


class PointVSF(BaseVSF):
    """A VSF model that consists of a set of particles.

    The stiffness field `stiffness` is defined at each particle.
    The axis_mode parameter defines the symmetry of the VSF material:
    - If axis_mode = 'isotropic', then the stiffness field is a scalar.
    - If axis_mode = 'axis_aligned', then the stiffness field is a 3-vector.
    - If axis_mode = 'anisotropic', then the stiffness field is a 3x3 matrix.

    Features can also be defined at each particle, which are stored in the
    `features` dictionary.  Typical features include:
    - 'volume': the volume of space represented by the particle
    - 'K_std': the standard deviation of the particle's stiffness
    - 'N_obs': the number of observations of the particle
    - 'mass': the mass of the particle
    - 'damping_ratio': the damping ratio of the particle
    - 'friction': the friction coefficient of the particle

    Attributes:
        rest_points (np.ndarray or torch.Tensor): Nx3 matrix giving rest
            positions of particles
        axis_mode (str): axis mode of the stiffness field
        stiffness (torch.Tensor): Nx1, Nx3, or Nx3x3 tensor giving the
            stiffness field at each particle
        features (dict[str,torch.Tensor]): optional features of the particles, each
            of leading dimension N.
    """

    def __init__(self,
                 rest_points: Union[np.ndarray,
                                    torch.Tensor] = None,
                 bbox: Optional[np.ndarray] = None,
                 voxel_size: Optional[float] = None,
                 axis_mode: str = 'isotropic',
                 features: Dict[str,
                                torch.Tensor] = None):
        if rest_points is None:
            # create a grid of points
            assert bbox is not None and voxel_size is not None
            assert isinstance(bbox, np.ndarray)
            assert bbox.shape == (2, 3)
            assert isinstance(voxel_size, float)
            bmin, bmax = bbox
            shape = [int(np.ceil((bmax[i] - bmin[i]) / voxel_size))
                     for i in range(3)]
            rest_points = torch.meshgrid(
                *[torch.linspace(bmin[i], bmax[i], shape[i]) for i in range(3)], indexing='ij')
            rest_points = [dim.reshape(-1) for dim in rest_points]
            rest_points = torch.stack(rest_points, dim=1)
            assert rest_points.shape[1] == 3

        if isinstance(rest_points, np.ndarray):
            rest_points = torch.from_numpy(rest_points)
        assert isinstance(
            rest_points, torch.Tensor), "Must provide a tensor as the rest points"
        self.rest_points = rest_points

        self.axis_mode = axis_mode
        assert axis_mode in ['isotropic', 'axis_aligned', 'anisotropic']
        if axis_mode == 'isotropic':
            self.stiffness = torch.zeros_like(rest_points[:, 0])
        elif axis_mode == 'axis_aligned':
            self.stiffness = torch.zeros_like(rest_points)
        elif axis_mode == 'anisotropic':
            self.stiffness = torch.zeros((len(rest_points), 3, 3))

        self.features = features if features is not None else {}
        assert isinstance(self.features, dict)

    @property
    def num_points(self) -> int:
        return len(self.rest_points)

    def getStiffness(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: interpolation")

    def getBBox(self) -> torch.Tensor:
        return torch.stack([self.rest_points.min(
            0)[0], self.rest_points.max(0)[0]], dim=0)

    def set_feature(self, name: str, feature: Union[np.ndarray, torch.Tensor]):
        """Sets / adds a feature to the VSF."""
        assert isinstance(name, str)
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature)
        assert isinstance(feature, torch.Tensor)
        assert feature.size(0) == len(self.rest_points)
        self.features[name] = feature.to(self.rest_points.device)

    def set_uniform_feature(
            self, name: str, value: Union[float, np.ndarray, torch.Tensor]):
        """Sets a uniform feature to all the VSF particles."""
        if isinstance(value, float):
            self.set_feature(name, torch.full((len(self.rest_points),), value))
            return
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        assert isinstance(value, torch.Tensor)
        self.features[name] = value.to(self.rest_points.device).reshape(
            1, -1).repeat(len(self.rest_points), 1)

    def feature_tensor(self, feature_names: List[str] = None) -> torch.Tensor:
        """Returns a tensor of features for all particles.

        If feature_names is None, returns all features.
        Otherwise, returns only the specified features.

        To avoid duplicate rest_points, when input feature_names contains 'rest_points',
        the function will return a tensor of rest_points instead of the feature tensor.

        Result has shape N x F where F is the sum of selected feature sizes.
        """
        if feature_names is None:
            feature_names = list(self.features.keys())
        N = len(self.rest_points)
        # return torch.stack([self.features[name].reshape(N,-1) for name in
        # feature_names], dim=1)
        features_list = []
        for name in feature_names:
            if name == 'rest_points':
                features_list.append(self.rest_points)
            else:
                assert name in self.features, f"Feature {name} not found"
                features_list.append(self.features[name].reshape(N, -1))
        return torch.stack(features_list, dim=1)
    
    def __getitem__(self, idx: Union[list, np.ndarray, torch.Tensor]) -> "PointVSF":
        """
        Return a subset of the PointVSF object corresponding to the provided indices.

        Args:
            idx (Union[list, np.ndarray, torch.Tensor]): Indices to subset the PointVSF. 
                Must be 1D and index along the primary point axis.

        Returns:
            PointVSF: A new PointVSF instance containing only the selected points.
        """
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        assert isinstance(idx, torch.Tensor)
        assert idx.ndim == 1
        assert idx.size(0) <= len(self.rest_points)
        new_rest_points = self.rest_points[idx, :]
        new_stiffness = self.stiffness[idx]
        new_features = {k: v[idx] for k, v in self.features.items()}
        vsf_subset = PointVSF(
            rest_points=new_rest_points,
            axis_mode=self.axis_mode,
            features=new_features
        )
        vsf_subset.stiffness = new_stiffness
        return vsf_subset

    def save(self, path: str):
        """Saves to either a folder or a Numpy archive (npz) file."""
        if path.endswith('/') or os.path.isdir(path):
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(
                os.path.join(
                    path,
                    'points.npy'),
                self.rest_points.cpu().numpy())
            np.save(os.path.join(path, 'K.npy'), self.stiffness.cpu().numpy())
            assert 'K' not in self.features
            assert 'points' not in self.features
            for k, v in self.features.items():
                np.save(os.path.join(path, k + '.npy'), v.cpu().numpy())
        else:
            save_dict = {
                "points": self.rest_points.cpu().numpy(),
                "axis_mode": self.axis_mode,
                "features": {
                    k: v.cpu().numpy() for (
                        k,
                        v) in self.features.items()},
                "K": self.stiffness.cpu().numpy()}
            np.savez(path, **save_dict)

    def load(self, path: str):
        """Loads from one of the standard file formats:

        - If path is a directory, then the following files are expected:
            - points.npy: Nx3 array of rest points
            - K.npy: Nx1, Nx3, or Nx3x3 array of stiffnesses
            - other .npy / .npz files: optional features
        - If path is a file, then it is a numpy file with the following fields:
            - points: Nx3 array of rest points
            - axis_mode: string
            - K: Nx1, Nx3, or Nx3x3 array of stiffnesses
            - features: dictionary of features
        """
        if os.path.isdir(path):
            if os.path.exists(
                os.path.join(
                    path,
                    'points.npy')) and os.path.exists(
                os.path.join(
                    path,
                    'K.npy')):
                points = np.load(os.path.join(path, 'points.npy'))
                stiffnesses = np.load(os.path.join(path, 'K.npy'))
                self.rest_points = torch.from_numpy(
                    points).to(self.rest_points.device)
                self.stiffness = torch.from_numpy(
                    stiffnesses).to(self.rest_points.device)
                for f in os.listdir(path):
                    if f.endswith('.npy') and f not in ['points.npy', 'K.npy']:
                        self.features[f[:-4]] = torch.from_numpy(
                            np.load(os.path.join(path, f))).to(self.rest_points.device)
                    if f.endswith('.npz'):  # assume sparse matrix
                        from scipy import sparse
                        data = sparse.load_npz(os.path.join(path, f))
                        dense = data.toarray()
                        self.features[f[:-4]] = torch.from_numpy(
                            dense).to(self.rest_points.device)
                self.axis_mode = 'isotropic' if self.stiffness.ndim == 1 else (
                    'axis_aligned' if self.stiffness.ndim == 2 else 'anisotropic')
            else:
                raise IOError("Invalid VSF directory")
        else:
            objs = np.load(path, allow_pickle=True)
            self.rest_points = torch.from_numpy(
                objs['points']).to(
                self.rest_points.device)
            # PointVSF may not have features
            if 'features' in objs.keys():
                features_dict: dict = objs['features'].item()
                for k, v in features_dict.items():
                    self.features[k] = torch.from_numpy(
                        v).to(self.rest_points.device)
            self.stiffness = torch.from_numpy(
                objs['K']).to(
                self.rest_points.device)
            default_axis_mode = 'isotropic' if self.stiffness.ndim == 1 else (
                'axis_aligned' if self.stiffness.ndim == 2 else 'anisotropic')
            self.axis_mode = objs.get('axis_mode', default_axis_mode)

        if len(self.stiffness) != len(self.rest_points):
            raise IOError("Invalid stiffness field size, {} != {}".format(
                len(self.stiffness), len(self.rest_points)))

    def to(self, device, dtype=None) -> PointVSF:
        """Converts the VSF to a given device or dtype"""
        
        new_rest_points = self.rest_points.to(device)        
        new_features = {
            k: v.to(device) for k, v in self.features.items()}

        if dtype is not None:
            new_rest_points = new_rest_points.to(dtype)
            for k, v in new_features.items():
                new_features[k] = v.to(dtype)

        return PointVSF(rest_points=new_rest_points, 
                        axis_mode=self.axis_mode, features=new_features)

    def compute_forces(self, point_indices: torch.LongTensor,
                       displaced_points: torch.Tensor) -> torch.Tensor:
        """Computes the forces on a set of points given their indices and displaced positions.

        All quantities are in the local frame of the VSF.

        Args:
            point_indices (torch.LongTensor): a length M tensor of point indices
            displaced_points (torch.Tensor): a Mx3 tensor of displaced points

        Returns:
            torch.Tensor: a Mx3 tensor of forces that would be applied to the VSF.
        """
        assert isinstance(point_indices, torch.Tensor)
        assert isinstance(displaced_points, torch.Tensor)
        assert displaced_points.ndim == 2 and displaced_points.size(1) == 3
        assert point_indices.ndim == 1 and point_indices.size(
            0) == displaced_points.size(0)
        stiffness = self.stiffness[point_indices]
        deformation = displaced_points - self.rest_points[point_indices]
        if self.axis_mode == 'isotropic':
            forces = stiffness.reshape(-1, 1) * deformation
        elif self.axis_mode == 'axis_aligned':
            forces = stiffness * deformation
        elif self.axis_mode == 'anisotropic':
            raise NotImplementedError("TODO: check this")
            forces = stiffness.dot(deformation)
        else:
            raise ValueError("Invalid VSF axis mode")
        return forces

    @property
    def device(self) -> torch.device:
        """Returns the device of the VSF."""
        return self.rest_points.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the VSF."""
        return self.rest_points.dtype