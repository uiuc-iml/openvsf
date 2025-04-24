import os
import shutil
import tempfile
import numpy as np
import torch
import pytest

from vsf.core.point_vsf import PointVSF, PointVSFConfig


def make_cube_points(n=10):
    """Generate a simple cube of 3D points."""
    return torch.stack(torch.meshgrid([
        torch.linspace(0, 1, n),
        torch.linspace(0, 1, n),
        torch.linspace(0, 1, n)
    ], indexing='ij'), dim=-1).reshape(-1, 3)


def test_init_with_rest_points():
    points = make_cube_points(5)
    vsf = PointVSF(rest_points=points, axis_mode='axis_aligned')
    assert isinstance(vsf.rest_points, torch.Tensor)
    assert vsf.rest_points.shape[1] == 3
    assert vsf.stiffness.shape == points.shape
    assert vsf.axis_mode == 'axis_aligned'
    assert isinstance(vsf.features, dict)


def test_init_with_bbox_voxel():
    bbox = np.array([[0, 0, 0], [1, 1, 1]])
    voxel_size = 0.5
    vsf = PointVSF(bbox=bbox, voxel_size=voxel_size)
    assert vsf.rest_points.ndim == 2 and vsf.rest_points.shape[1] == 3


def test_invalid_axis_mode():
    points = make_cube_points(3)
    with pytest.raises(AssertionError):
        PointVSF(rest_points=points, axis_mode='invalid')


def test_num_points_property():
    points = make_cube_points(4)
    vsf = PointVSF(rest_points=points)
    assert vsf.num_points == points.shape[0]


def test_set_and_get_feature():
    points = make_cube_points(4)
    vsf = PointVSF(rest_points=points)
    feature = torch.arange(points.shape[0], dtype=torch.float64)
    vsf.set_feature('volume', feature)
    assert 'volume' in vsf.features
    assert torch.allclose(vsf.features['volume'], feature)


def test_set_uniform_feature():
    points = make_cube_points(4)
    vsf = PointVSF(rest_points=points)
    vsf.set_uniform_feature('rand_feat', 1.5)
    assert 'rand_feat' in vsf.features
    assert torch.allclose(vsf.features['rand_feat'], torch.full((vsf.num_points,), 1.5))


def test_feature_tensor():
    points = make_cube_points(4)
    vsf = PointVSF(rest_points=points)
    vsf.set_uniform_feature('mass', 1.0)
    vsf.set_uniform_feature('volume', 2.0)
    f_tensor = vsf.feature_tensor(['mass', 'volume'])
    assert f_tensor.shape == (vsf.num_points, 2, 1)


def test_getBBox():
    points = torch.tensor([[0, 0, 0], [1, 2, 3], [0.5, 0.5, 0.5]])
    vsf = PointVSF(rest_points=points)
    bbox = vsf.getBBox()
    assert bbox.shape == (2, 3)
    assert torch.allclose(bbox[0], torch.tensor([0., 0., 0.]))
    assert torch.allclose(bbox[1], torch.tensor([1., 2., 3.]))


def test_to_device():
    points = make_cube_points(3)
    vsf = PointVSF(rest_points=points)
    vsf.set_uniform_feature('mass', 1.0)
    vsf_cuda = vsf.to(torch.device('cpu'))  # switch to 'cuda' if GPU is available
    assert vsf_cuda.rest_points.device == torch.device('cpu')
    assert 'mass' in vsf_cuda.features


def test_compute_forces_isotropic():
    points = make_cube_points(2)
    vsf = PointVSF(rest_points=points, axis_mode='isotropic')
    vsf.stiffness[:] = 2.0
    indices = torch.tensor([0, 1], dtype=torch.long)
    displaced = vsf.rest_points[indices] + torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])
    forces = vsf.compute_forces(indices, displaced)
    expected = 2.0 * (displaced - vsf.rest_points[indices])
    assert torch.allclose(forces, expected)


def test_compute_forces_axis_aligned():
    points = make_cube_points(2)
    vsf = PointVSF(rest_points=points, axis_mode='axis_aligned')
    vsf.stiffness[:] = 2.0
    indices = torch.tensor([0, 1], dtype=torch.long)
    displaced = vsf.rest_points[indices] + 0.1
    forces = vsf.compute_forces(indices, displaced)
    expected = 2.0 * (displaced - vsf.rest_points[indices])
    assert torch.allclose(forces, expected)


def test_save_and_load_folder(tmp_path):
    points = make_cube_points(2)
    vsf = PointVSF(rest_points=points, axis_mode='isotropic')
    vsf.set_uniform_feature('volume', 1.0)
    save_dir = tmp_path / "vsf_save"
    save_dir.mkdir(parents=True, exist_ok=True)  
    vsf.save(str(save_dir))
    
    assert (save_dir / 'points.npy').exists()
    assert (save_dir / 'K.npy').exists()

    loaded = PointVSF(rest_points=points)
    loaded.load(str(save_dir))
    assert torch.allclose(loaded.rest_points, vsf.rest_points)
    assert torch.allclose(loaded.stiffness, vsf.stiffness)
    assert 'volume' in loaded.features
    assert torch.allclose(loaded.features['volume'], vsf.features['volume'])


def test_save_and_load_npz(tmp_path):
    points = make_cube_points(2)
    vsf = PointVSF(rest_points=points, axis_mode='axis_aligned')
    vsf.set_uniform_feature('mass', 0.5)
    save_path = tmp_path / "vsf.npz"
    vsf.save(str(save_path))

    loaded = PointVSF(rest_points=points)
    loaded.load(str(save_path))
    assert torch.allclose(loaded.rest_points, vsf.rest_points)
    assert torch.allclose(loaded.stiffness, vsf.stiffness)
    assert loaded.axis_mode == vsf.axis_mode
    assert 'mass' in loaded.features


def test_config_dataclass():
    cfg = PointVSFConfig(voxel_size=0.1, axis_mode='anisotropic')
    assert isinstance(cfg, PointVSFConfig)
    assert cfg.voxel_size == 0.1
    assert cfg.axis_mode == 'anisotropic'
    

def test_subset_indices():
    points = make_cube_points(3)
    features = {'mass': torch.tensor([1.0, 2.0, 3.0])}
    vsf = PointVSF(rest_points=points, axis_mode='axis_aligned', features=features)
    indices = torch.tensor([0, 2], dtype=torch.long)
    vsf_subset = vsf[indices]
    assert vsf_subset.rest_points.shape[0] == 2
    assert torch.allclose(vsf_subset.rest_points, points[indices, :])
    assert torch.allclose(vsf_subset.stiffness, vsf.stiffness[indices])
    assert torch.allclose(vsf_subset.features['mass'], features['mass'][indices])
