import numpy as np
import pyvista as pv

def point_cloud_to_pyvista_volume(points : np.ndarray, colors : np.ndarray, 
                                  voxel_size=0.1, opacity_scaling=1.0) -> pv.ImageData:
    """
    Use PyVista to create a volume rendering of a point cloud.
    
    TODO: this function is not working properly. The colors are not being rendered correctly.
    """
    # # Ensure colors are in [0, 1]
    # if np.issubdtype(colors.dtype, np.integer) or colors.max() > 1.0:
    #     colors = colors.astype(np.float32) / 255.0

    # Calculate grid bounds
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    # Calculate number of voxels in each dimension
    x_voxels = int(np.ceil((x_max - x_min) / voxel_size))
    y_voxels = int(np.ceil((y_max - y_min) / voxel_size))
    z_voxels = int(np.ceil((z_max - z_min) / voxel_size))
    
    # Initialize arrays to accumulate color and count
    voxel_rgba = np.zeros((x_voxels, y_voxels, z_voxels, 4), dtype=np.float32)
    voxel_count = np.zeros((x_voxels, y_voxels, z_voxels), dtype=np.int32)

    print('voxel count shape:', voxel_count.shape)

    # Calculate voxel indices for each point
    voxel_i = np.floor((points[:, 0] - x_min) / voxel_size).astype(int)
    voxel_j = np.floor((points[:, 1] - y_min) / voxel_size).astype(int)
    voxel_k = np.floor((points[:, 2] - z_min) / voxel_size).astype(int)

    # Filter out points outside the grid
    valid = (voxel_i >= 0) & (voxel_i < x_voxels) & \
            (voxel_j >= 0) & (voxel_j < y_voxels) & \
            (voxel_k >= 0) & (voxel_k < z_voxels)

    voxel_i = voxel_i[valid]
    voxel_j = voxel_j[valid]
    voxel_k = voxel_k[valid]
    colors = colors[valid]

    # Accumulate colors and counts
    np.add.at(voxel_rgba, (voxel_i, voxel_j, voxel_k, 0), colors[:, 0])
    np.add.at(voxel_rgba, (voxel_i, voxel_j, voxel_k, 1), colors[:, 1])
    np.add.at(voxel_rgba, (voxel_i, voxel_j, voxel_k, 2), colors[:, 2])
    np.add.at(voxel_count, (voxel_i, voxel_j, voxel_k), 1)

    # Compute average colors where there are points
    mask = voxel_count > 0
    # for c in range(3):
    #     voxel_rgba[..., c][mask] /= voxel_count[mask]

    # Compute alpha (normalized density)
    if np.any(mask):
        max_count = voxel_count.max()
        # voxel_rgba[..., 3] = (voxel_count / max_count) * opacity_scaling
        voxel_rgba[..., 3] = 0.1
        # voxel_rgba[voxel_count > 0, 3] = 1
    else:
        voxel_rgba[..., 3] = 0

    # Create a grid with points at voxel centers
    new_origin = (
        x_min + 0.5 * voxel_size,
        y_min + 0.5 * voxel_size,
        z_min + 0.5 * voxel_size,
    )
    grid = pv.ImageData(
        dimensions=(x_voxels, y_voxels, z_voxels),
        origin=new_origin,
        spacing=(voxel_size, voxel_size, voxel_size)
    )

    # convert voxel_rgba to uint8
    voxel_rgba = (voxel_rgba * 255).astype(np.uint8)

    # Assign RGBA data to point data
    grid.point_data['rgba'] = voxel_rgba.reshape(-1, 4)

    return grid
