
import open3d as o3d
from pathlib import Path
import os
from .common import stiffness_to_color, colorize, autodetect_stiffness_values, O3DViewParams
import torch
import numpy as np
from .. import NeuralVSF, PointVSF
from dataclasses import dataclass, field, asdict
from klampt import WorldModel
from typing import List, Tuple, Union, Optional


def colorize_contacts(contact_pcd, world_pts, world_nms, contact_rest_pts):
    """
    Colorize the contact point cloud based on the contact status.
    
    Args:
        contact_pcd: Open3D point cloud object
        world_pts: Nx3 numpy array of world points
        world_nms: Nx3 numpy array of world normals
        contact_rest_pts: Nx3 numpy array of contact rest points
    """
    dr = world_pts - contact_rest_pts
    dr_dot_nm = (dr*world_nms).sum(axis=1)

    pcd_color_ary = np.zeros(world_pts.shape)
    pcd_color_ary[dr_dot_nm>0, :] = [0.1, 0.9, 0.1] # attached, green
    pcd_color_ary[dr_dot_nm<0, :] = [0.9, 0.1, 0.1] # deteached, red

    # down scale normals
    # world_nms *= 0.1

    # rest_contact_pts = obj_sim.rest_pts[obj_sim.contact_pts_idx, :]
    assert world_pts.shape == world_nms.shape
    assert world_nms.shape == pcd_color_ary.shape
    contact_pcd.points  = o3d.utility.Vector3dVector(world_pts)
    contact_pcd.normals = o3d.utility.Vector3dVector(world_nms)
    contact_pcd.colors  = o3d.utility.Vector3dVector(pcd_color_ary)
    # contact_pcd.paint_uniform_color([0.1, 0.9, 0.1])

def create_o3dMesh_ball(radius=0.01, color=[1, 0, 0], center=None):
    """Create a Open3d sphere mesh with the given radius and color."""
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    ball.compute_vertex_normals()
    ball.paint_uniform_color(color)
    if center is not None:
        ball.translate(center)
    return ball

def create_o3dMesh_box(w, h, d, color=[0.2, 0.9, 0.2]):
    """Create a Open3d box mesh with the given width, height, depth and color."""
    box = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.paint_uniform_color(color)
    return box

def calculate_zy_rotation_for_arrow(vec):
    """Utility function to calculate the rotation matrix for the arrow."""
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def create_vector_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=[0.707, 0.707, 0.0]):
    """
    Create an arrow mesh from the origin to the end point.
    
    TODO: find better way to specify parameters of an arrow
    """
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    # work for neural vsf estimation
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        # cone_radius=size/17.5 * scale,
        cone_radius=0.03/17.5 * scale,
        cone_height=0.10/17.5 * scale,
        # cylinder_radius=size/30 * scale,
        cylinder_radius=0.03/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    # mesh = o3d.geometry.TriangleMesh.create_arrow(
    #     # cone_radius=size/17.5 * scale,
    #     cone_radius=0.15/17.5 * scale,
    #     cone_height=0.45/17.5 * scale,
    #     # cylinder_radius=size/30 * scale,
    #     cylinder_radius=0.15/30 * scale,
    #     cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return(mesh)

def create_arrow_lst(p1_ary, p2_ary, min_size=0.01, **args):
    """Create a list of arrows from p1 to p2."""
    arrow_lst = []
    for p1, p2 in zip(p1_ary, p2_ary):
        if np.linalg.norm(p2-p1) > min_size:
            arrow_lst.append(create_vector_arrow(p2, origin=p1, **args))
    return arrow_lst

def pick_pcd_pts(pcd):
    """ 
    1) Please pick at least three correspondences using [shift + left click]
       Press [shift + right click] to undo point picking
    2) After picking points, press 'Q' to close the window
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json('data/render_params.json')
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

def NonEmptyPC(mode='zero'):
    """
    Utility function to create a non-empty point cloud.
    
    This function is used because Open3D visualizer needs
    pcd initially to be non-empty to display it.
    """
    vis_pcd = o3d.geometry.PointCloud()
    if mode == 'rand':
        points = np.random.uniform(low=0, high=1, size=(100, 3))
        colors = np.random.uniform(low=0, high=1, size=(100, 3))
    elif mode == 'zero':
        points = np.zeros((1, 3))
        colors = np.zeros((1, 3))
    vis_pcd.points = o3d.utility.Vector3dVector(points)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    return vis_pcd

def create_motion_lines(prev_pts: np.ndarray, curr_pts: np.ndarray, 
                        return_pcd: bool=False) -> Union[o3d.geometry.LineSet, Tuple[o3d.geometry.PointCloud]]:
    """
    Create a line set between the previous and current points.
    
    Args:
        prev_pts: Nx3 numpy array of previous points
        curr_pts: Nx3 numpy array of current points
        return_pcd: If True, return the point clouds as well.

    Returns:
        line_set: Open3D line set object
    """
    assert(prev_pts.shape == curr_pts.shape)
    prev_pcd = o3d.geometry.PointCloud()
    prev_pcd.points = o3d.utility.Vector3dVector(prev_pts)
    prev_pcd.paint_uniform_color([0, 0, 1])

    curr_pcd = o3d.geometry.PointCloud()
    curr_pcd.points = o3d.utility.Vector3dVector(curr_pts)
    curr_pcd.paint_uniform_color([1, 0, 0])

    pcd_correspondence = [[i, i] for i in range(curr_pts.shape[0])]
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(prev_pcd, curr_pcd, pcd_correspondence)
    if return_pcd:
        return prev_pcd, curr_pcd, line_set
    else:
        return line_set



def vis_contact(arm_pts, curr_contact_pts, new_contact_pts):
    curr_pcd = o3d.geometry.PointCloud()
    curr_pcd.points = o3d.utility.Vector3dVector(curr_contact_pts)
    curr_pcd.paint_uniform_color([0.9, 0.1, 0.1])

    arm_pcd = o3d.geometry.PointCloud()
    arm_pcd.points = o3d.utility.Vector3dVector(arm_pts)
    arm_pcd.paint_uniform_color([0.1, 0.9, 0.1])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_contact_pts)
    new_pcd.paint_uniform_color([0.1, 0.1, 0.9])
    o3d.visualization.draw_geometries([arm_pcd, new_pcd, curr_pcd])


class VisManager:
    """
    A helper class to manage Open3D visualization.
    
    Args:
        fig_dir: Directory to save the figures
        enable_vis: If True, enable visualization
        b_min: Minimum bounding box coordinates, [x_min, y_min, z_min]
        b_max: Maximum bounding box coordinates, [x_max, y_max, z_max]
        bbox_color: Bounding box color
        background_color: Background color
    """
    def __init__(self, fig_dir=None, enable_vis=True,
                 b_min=[-1, -1, -1], b_max=[1, 1, 1], bbox_color=None,
                 background_color=None):

        self.enable_vis = enable_vis
        if enable_vis:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window()
            bbox = o3d.geometry.AxisAlignedBoundingBox(b_min, b_max)
            if bbox_color is not None:
                bbox.color = bbox_color
            self.o3d_vis.add_geometry(bbox)
            self.o3d_vis.remove_geometry(bbox, reset_bounding_box=False)
        else:
            self.o3d_vis = None

        self.pcd_lst = []

        if fig_dir is not None:
            self.set_fig_dir(fig_dir)

        if background_color is not None:
            self.set_background_color(background_color)
    
    def set_fig_dir(self, fig_dir):
        """Set the directory to save the figures."""
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        self.fig_dir = fig_dir

    def init_pcd_lst(self, num_pcd):
        """Initialize the point cloud list."""
        for _ in range(num_pcd):
            self.add_pcd()

    def add_pcd(self, pcd=None, reset_bbox=True):
        """Return index of pcd in the list"""
        if pcd is None:
            pcd = NonEmptyPC()
        if self.enable_vis:
            self.o3d_vis.add_geometry(pcd, reset_bounding_box=reset_bbox)
        pcd_idx = len(self.pcd_lst)
        self.pcd_lst.append(pcd)
        return pcd_idx
    
    def extend_pcd_lst(self, num_pcd=None, pcd_lst=None):
        """
        Extend the point cloud list.
        
        If num_pcd is not None, create num_pcd point clouds.
        If pcd_lst is not None, add the given point clouds to the list.
        """
        if num_pcd is not None:
            assert pcd_lst is None
            pcd_lst = [ NonEmptyPC() for _ in range(num_pcd) ]
        pcd_idx_lst = []
        for pcd in pcd_lst:
            pcd_idx = self.add_pcd(pcd)
            pcd_idx_lst.append(pcd_idx)
        return pcd_idx_lst

    def refresh_display(self):
        """Refresh the Open3D display."""
        if self.enable_vis:
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer() 

    def update_all(self, fig_fn:str=None):
        """
        Update all the geometries and save the figure.
        
        When fig_fn is not None, save the figure in the fig_dir.
        """
        if not self.enable_vis:
            return

        for pcd in self.pcd_lst:
            self.o3d_vis.update_geometry(pcd)

        self.refresh_display()
        if fig_fn is not None:
            self.save_fig(fig_fn)
    
    def save_fig(self, fig_fn: str):
        """
        Save the figure in the fig_dir.
        """
        if self.enable_vis:
            fig_path = os.path.join(self.fig_dir, fig_fn)
            self.o3d_vis.capture_screen_image(fig_path)
    
    def update_pcd(self, pcd_idx, pts=None, colors=None, normals=None):
        """
        Update the point cloud with the given index.
        
        Args:
            pcd_idx: Index of the point cloud
            pts: Nx3 numpy array of points
            colors: Nx3 numpy array of colors
            normals: Nx3 numpy array of normals
        """
        assert pcd_idx < len(self.pcd_lst)

        pcd = self.pcd_lst[pcd_idx]
        if pts is not None:
            pts = np.array(pts)
            assert pts.ndim == 2 and pts.shape[1] == 3
            pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            colors = np.array(colors)
            if colors.ndim != 2:
                pcd.paint_uniform_color(colors)
            else:
                assert np.array(pcd.points).shape == colors.shape
                pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        if self.enable_vis:
            self.o3d_vis.update_geometry(pcd)
            self.refresh_display()

    def set_view_params(self, view_params : O3DViewParams):
        """Set the view parameters for the visualization."""
        view_params.set(self.o3d_vis)

    def set_render_option(self, render_params_fn=None):
        """Set the render options for the visualization."""
        self.o3d_vis.get_render_option().load_from_json(render_params_fn)
    
    def set_background_color(self, color:np.ndarray):
        """Set the background color for the visualization."""
        opt = self.o3d_vis.get_render_option()
        opt.background_color = color

    def vis_show_normal(self, point_show_normal):
        """Set the visualization to show normals."""
        self.o3d_vis.get_render_option().point_show_normal = point_show_normal


def vsf_to_point_cloud(vsf : Union[PointVSF,NeuralVSF],
                       mask : Optional[Union[str,np.ndarray]]='auto',
                       N_samples = 200000,
                       masked_view_fraction = 0.035,
                       auto_stiffness_threshold = 0.01,
                       feature : str = None, feature_idx : Union[int,Tuple[int]]= 0,
                       cmap = 'YlOrRd') -> o3d.geometry.PointCloud:
    """Converts a VSF to an Open3D point cloud for visualization.
    
    Args:
        vsf: a VSF model
        mask: a boolean mask of size N_samples, None (for no mask), or 'auto' for
            automatic masking based on stiffness values.
        N_samples: the number of samples to use for visualization, for NeuralVSF objects.
        masked_view_fraction: the fraction of the masked points to show
        auto_stiffness_threshold: the threshold for automatic masking of stiffness values.
        feature: the feature to visualize, for PointVSF objects.
        feature_idx: the index of the feature to visualize, for PointVSF objects.
        cmap: the colormap to use for visualizing the feature.  Can also be 'random'
    """
    colors = None
    if isinstance(vsf, NeuralVSF):
        vsf_samples = torch.rand(N_samples, 3, device=vsf.device) * (vsf.vsfNetwork.aabb[1] - vsf.vsfNetwork.aabb[0]) + vsf.vsfNetwork.aabb[0]
        vsf_stiffness = vsf.vsfNetwork(vsf_samples).detach().cpu().numpy().squeeze()
        vsf_samples = vsf_samples.cpu().numpy()
    elif isinstance(vsf, PointVSF):
        vsf_samples = vsf.rest_points.cpu().numpy()
        if feature is not None:
            vsf_stiffness = vsf.features[feature]
            if vsf_stiffness.ndim == 2:
                if isinstance(feature_idx,(tuple,list)):
                    colors = [vsf_stiffness[:,i] for i in feature_idx]
                    while len(colors) < 3:
                        colors.append(np.zeros(len(vsf_stiffness)))
                    #normalize colors to [0,1] and convert to Nx3
                    colors = [np.clip((c - np.min(c)) / (np.max(c) - np.min(c)),0,1) for c in colors]
                    colors = np.array(colors).T
                else:
                    colors = colorize(vsf_stiffness[:, feature_idx],cmap)
        else:
            vsf_stiffness = vsf.stiffness
        vsf_stiffness = vsf_stiffness.cpu().numpy()
    else:
        raise ValueError("Unknown VSF type")
    if colors is None:
        print("Stiffness range: ", np.min(vsf_stiffness), np.max(vsf_stiffness))
        colors = stiffness_to_color(vsf_stiffness,cmap=cmap)

    if isinstance(mask,str) and mask == 'auto':
        # mask = vsf_stiffness > np.percentile(vsf_stiffness[vsf_stiffness>auto_stiffness_threshold], 40)
        mask = vsf_stiffness > auto_stiffness_threshold

    if mask is not None:
        colors[~mask] = [0.7, 0.7, 0.7]
        mask[~mask] = np.random.rand(len(mask[~mask])) < masked_view_fraction
        vsf_samples = vsf_samples[mask]
        vsf_stiffness = vsf_stiffness[mask]
        colors = colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vsf_samples)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def vsf_show(vsf : Union[PointVSF,NeuralVSF],
             mask : Optional[Union[str,np.ndarray]]='auto',
             N_samples = 200000,
             masked_view_fraction = 0.035,
             auto_stiffness_threshold = 0.01,
             view_params : O3DViewParams = O3DViewParams(), 
             vis_time : float = None) -> VisManager:
    """Visualizes a VSF using Open3D.
    
    Args:
        vsf: a VSF model
        mask: a boolean mask of size N_samples, None (for no mask), or 'auto' for
            automatic masking based on stiffness values.
        N_samples: the number of samples to use for visualization, for NeuralVSF objects.
        masked_view_fraction: the fraction of the masked points to show
        auto_stiffness_threshold: the threshold for automatic masking of stiffness values.
        view_params: the view parameters for the visualization.
        vis_time: the time to wait before closing the visualization, default is None
            to indicate that the visualization should not be closed automatically, 
            when the variable is set to positive float, the visualization will be closed
            after the given time.
    """
    pcd = vsf_to_point_cloud(vsf, mask, N_samples, masked_view_fraction, auto_stiffness_threshold)
    from klampt.io import open3d_convert
    pcd = open3d_convert.from_open3d(pcd)
    bbox = vsf.getBBox().cpu().numpy().astype(float)
    vis = VisManager(enable_vis=True, b_min=bbox[0], b_max=bbox[1])
    vis.add_pcd(pcd)
    vis.set_view_params(view_params)
    return vis
    

def world_o3d_geometries(world : WorldModel) -> List[o3d.geometry.Geometry]:
    """
    Get the open3d geometries of the objects in the Klamp't world
    """
    from klampt.io import open3d_convert
    from klampt.math import se3    

    o3d_geom_lst = []
    for geom_idx in range(world.numRigidObjects()):
        geom = world.rigidObject(geom_idx).geometry()
        if geom.type() == 'TriangleMesh':
            o3d_geom = open3d_convert.to_open3d(geom.getTriangleMesh())
            o3d_geom.compute_vertex_normals()
        elif geom.type() == 'ImplicitSurface':
            o3d_geom = open3d_convert.to_open3d(geom.getImplicitSurface())
        elif geom.type() == 'PointCloud':
            o3d_geom = open3d_convert.to_open3d(geom.getPointCloud())
        else:
            raise ValueError("Unsupported geometry type")

        Hmat = se3.ndarray(geom.getCurrentTransform())
        o3d_geom.transform(Hmat)
        o3d_geom_lst.append(o3d_geom)
    return o3d_geom_lst
