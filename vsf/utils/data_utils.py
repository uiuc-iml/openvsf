from ..dataset.base_dataset import BaseDataset
import os
import yaml
import numpy as np
import torch
import open3d as o3d
from typing import Union,List,Tuple

# recursive function to convert all elements of a list/tuple/dict to torch.tensor
def convert_to_tensor(data : Union[list,tuple,dict,np.ndarray,torch.Tensor], device='cpu') -> Union[list,tuple,dict,torch.Tensor]:
    if isinstance(data, list):
        return [convert_to_tensor(d, device=device) for d in data]
    elif isinstance(data, tuple):
        return tuple([convert_to_tensor(d, device=device) for d in data])
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, device=device) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float64 or data.dtype == np.object_:
            data = data.astype(np.float32)
        return torch.tensor(data, device=device)
    return data

# recursive function to convert all elements of a list/tuple/dict to numpy.ndarray
def convert_to_numpy(data : Union[list,tuple,dict,np.ndarray,torch.Tensor]) -> Union[list,tuple,dict,np.ndarray]:
    if isinstance(data, list):
        return [convert_to_numpy(d) for d in data]
    elif isinstance(data, tuple):
        return tuple([convert_to_numpy(d) for d in data])
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data

def to_json_dict(obj : dict) -> dict:
    """Converts all numpy arrays and torch Tensors to lists in
    a list / dictionary to prepare for saving to JSON / YAML."""
    if isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (tuple,list)):
        return [to_json_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj

def remap_dict(data, *key_mappings) -> List[dict]:
    """Remaps the keys of a dictionary according to the provided mappings.

    Example:
        data = {'old_key1': 'value1', 'old_key2': 'value2'}
        remapped_data = remap_dict(data, {'new_key1': 'old_key1', 'new_key2': 'old_key2'})
        print(remapped_data)  # {'new_key1': 'value1', 'new_key2': 'value2'}

    Args:
        data (dict): The dictionary to remap.
        key_mappings (list of dict): The key mappings, where each tuple
            contains a mapping of old keys to new keys.
            For example, [{'old_key1': 'new_key1', 'old_key2': 'new_key2'}].

    Returns:
        list of dict: The remapped dictionaries.
    """
    res = []
    for km in key_mappings:
        data_selected = {}
        for k, v in km.items():
            if v in data:
                data_selected[k] = data[v]
        res.append(data_selected)
    return res

def remap_dict_in_seq(sequence_data, *key_mappings) -> List[List[dict]]:
    """Remaps the keys of a sequence of dictionaries according to the provided mappings.
    
    Example:
        sequence_data = [{'old_key1': 'value1', 'old_key2': 'value2'},
                         {'old_key1': 'value3', 'old_key2': 'value4'}]
        remapped_sequence = remap_dict_in_seq(sequence_data, {'new_key1': 'old_key1', 'new_key2': 'old_key2'})
        print(remapped_sequence)  # [{'new_key1': 'value1', 'new_key2': 'value2'}, {'new_key1': 'value3', 'new_key2': 'value4'}].

    Args:
        sequence_data (list of dict): The sequence of dictionaries to remap.
        key_mappings (list of dict): The key mappings, where each tuple
            contains a mapping of old keys to new keys.
            For example, [{'old_key1': 'new_key1', 'old_key2': 'new_key2'}].

    Returns:
        list of list of dict: The remapped dictionaries list.
    """
    res = []
    for km in key_mappings:
        res2 = []
        for data in sequence_data:
            res2.append(remap_dict(data, km)[0])
        res.append(res2)
    return res
    

def index_to_coord(index, b_min, b_max, vg_shape):
    b_min, b_max, vg_shape = np.array(b_min), np.array(b_max), np.array(vg_shape)
    coord = (index / vg_shape) * (b_max - b_min) + b_min
    return coord

def coord_to_index(coord, b_min, b_max, vg_shape, 
                   unique=False, format='npARY'):
    b_min, b_max, vg_shape = np.array(b_min), np.array(b_max), np.array(vg_shape)
    g_size = (b_max-b_min) / vg_shape
    # index = np.floor((coord-b_min) / g_size)
    index = np.rint((coord-b_min) / g_size)
    index = index.astype(dtype=np.int)

    if unique:
        index = np.unique(index, axis=0)

    if format == 'npARY':
        return index
    elif format == 'list':
        return index[:, 0], index[:, 1], index[:, 2]

def transform_points(coord:Union[np.ndarray,torch.Tensor], H:np.ndarray, rigid=True) -> Union[np.ndarray,torch.Tensor]:
    """Transforms a set of points by a 4x4 transformation matrix.
    
    If rigid is True, only the rotation and translation components of the
    transformation matrix are used. Otherwise, the full 4x4 matrix is used.

    Args:
        coord (np.ndarray or torch.Tensor): Nx3 array of points
        H (np.ndarray): 4x4 transformation matrix
        rigid (bool): If True, only the rotation and translation components
            of the transformation matrix are used.
    
    Returns:
        np.ndarray or torch.Tensor: Nx3 array of transformed points
    """
    assert coord.shape[1] == 3
    assert H.shape == (4, 4)
    if isinstance(coord, torch.Tensor):
        if not isinstance(H, torch.Tensor):
            H = torch.tensor(H, device=coord.device, dtype=coord.dtype)
        if rigid:
            return coord @ H[:3, :3].T + H[:3, 3]
        else:
            num_pts = coord.shape[0]
            coord_h = torch.cat((coord, torch.ones((num_pts,1), device=coord.device, dtype=coord.dtype)),
                                dim=1)
            transed_coord_h = coord @ H.T
            return transed_coord_h[:,:3]
    else:
        if rigid:
            return coord.dot(H[:3, :3].T) + H[:3, 3]
        else:
            coord_h = np.append(coord, np.ones((coord.shape[0],1)), axis=1)
            transed_coord_h = coord_h @ H.T
            return transed_coord_h[:,:3]
        
def transform_directions(dirs:Union[np.ndarray,torch.Tensor], H:np.ndarray, rigid=True) -> Union[np.ndarray,torch.Tensor]:
    """Transforms a set of normals by a 4x4 transformation matrix.
    
    If rigid is True, only the rotation component of the
    transformation matrix is used. Otherwise, the full 4x4 matrix is used.
    
    Args:
        dirs (np.ndarray or torch.Tensor): Nx3 array of normals
        H (np.ndarray): 4x4 transformation matrix
        rigid (bool): If True, only the rotation component
            of the transformation matrix is used.
    
    Returns:
        np.ndarray or torch.Tensor: Nx3 array of transformed points
    """
    if isinstance(dirs, torch.Tensor):
        if not isinstance(H, torch.Tensor):
            H = torch.tensor(H, device=dirs.device, dtype=dirs.dtype)
        if rigid:
            # TODO: temprorary fix, ideally device should propagate through the whole simulation
            H = H.to(device=dirs.device, dtype=dirs.dtype)
            return dirs @ H[:3, :3].T
        else:
            num_pts = dirs.shape[0]
            dirs_h = torch.cat((dirs, torch.zeros((num_pts,1), device=dirs.device, dtype=dirs.dtype)),
                                dim=1)
            transed_dirs_h = dirs @ H.T
            return transed_dirs_h[:,:3]
    else:
        if rigid:
            return dirs.dot(H[:3, :3].T)
        else:
            dirs_h = np.append(dirs, np.zeros((dirs.shape[0],1)), axis=1)
            transed_dirs_h = dirs_h @ H.T
            return transed_dirs_h[:,:3]

def get_AABB(dataset : BaseDataset, point_idx:str, eps=1e-2):
    """Assumes the dataset sequence returns a dictionary for
    each time step with the key `point_idx` containing the points
    you would like to extract"""
    points = []
    for i in range(len(dataset)):
        traj = dataset[i]
        for j in range(len(traj)):
            points.append(np.array(traj[j][point_idx], dtype=np.float32))

    points = np.concatenate(points, axis=0)
    b_min = np.min(points, axis=0) - eps
    b_max = np.max(points, axis=0) + eps
    return b_min, b_max


def sdf_to_points(sdf:np.ndarray, aabb_min:np.ndarray, aabb_max:np.ndarray, thresh:float):
    """Extracts 3D coordinates from a signed distance field (SDF) grid
    where the SDF value is less than a threshold."""
    grid_size = sdf.shape[0]
    assert sdf.shape == (grid_size, grid_size, grid_size)
    
    # Generate voxel grid coordinates
    x = np.linspace(aabb_min[0], aabb_max[0], grid_size)
    y = np.linspace(aabb_min[1], aabb_max[1], grid_size)
    z = np.linspace(aabb_min[2], aabb_max[2], grid_size)
    
    # Create full grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # shape (grid_size, grid_size, grid_size)
    
    # Stack into shape (grid_size, grid_size, grid_size, 3)
    points = np.stack([xx, yy, zz], axis=-1)
    
    # Mask where sdf < threshold
    mask = sdf < thresh
    
    # Extract 3D coordinates
    selected_points = points[mask]  # shape (N, 3)
    
    return selected_points

def cone_sample_dir(cone_dir=np.array([0.0, 1.0, 0.0]), cone_angle=np.pi/3):
    cone_dir /= np.linalg.norm(cone_dir)

    rand_v = np.random.randn(3)
    rand_v -= cone_dir*np.dot(rand_v, cone_dir)
    rand_v /= np.linalg.norm(rand_v)

    assert np.allclose(np.dot(rand_v, cone_dir), 0.0)

    theta = np.random.uniform(-cone_angle, cone_angle)
    d = rand_v * np.sin(theta) + cone_dir * np.cos(theta)

    assert np.allclose(np.linalg.norm(d), 1.0)
    return d

def clean_light_points(pcd, light_min=0.0, light_max=0.6, verbose=False):
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    cmin, cmax = colors.min(axis=1), colors.max(axis=1)
    lightness = (cmin + cmax) / 2.0

    mask = (lightness > light_min) & (lightness < light_max)

    # colors[mask, :] = [0.0, 1.0, 0.0]
    # colors[~mask, :] = [1.0, 0.0, 0.0]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    points = points[mask, :]
    colors = colors[mask, :]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)
    return new_pcd