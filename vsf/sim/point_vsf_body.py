from klampt.math import se3,so3
from .klampt_world_wrapper import klamptWorldWrapper
from klampt.model.geometry import TriangleMesh, PointCloud
from ..utils.perf import PerfRecorder, DummyRecorder
from ..utils.data_utils import transform_points,transform_directions
import numpy as np
import torch
import open3d as o3d
from .. import PointVSF


from dataclasses import dataclass
from typing import Dict,Tuple,List,Union



@dataclass
class ContactParams:
    friction_mu: float = 1.0
    sim_slide: bool = True
    slide_step: float = 0.01
    geometry_padding1: float = 0.003
    geometry_padding2: float = 0.003

class PointVSFQuasistaticSimBody:
    """Stores information to simulate quasistatic interaction with a point VSF.

    A stick-slip model is used to simulate sliding contacts.  Each point that
    penetrates another object first gets associated with an anchor point on that
    object. 

    Attributes:
        vsf_model: The point VSF model.
        pose: The current pose of the body.
        contact_params: The contact parameters.
        perfer: The performance recorder.
        verbose: Whether to print debug information.
        obj_knn: The nearest neighbor search for the object.
        point_object_idx: A boolean array indicating which VSF points are in contact.
        point_local: The deformed position of each point in the VSF, in VSF local coordinates.
        anchor_local: The anchor position of each point in the VSF, in other-object local coordinates.
        anchor_normal_local: The anchor normal of the contacted location, pointing outward, in other-object local coordinates.
    """
    def __init__(self, vsf_model:PointVSF, contact_params:ContactParams, 
                 triangle_mesh: Union[TriangleMesh, o3d.geometry.TriangleMesh]=None) -> None:
        """
        Initialize the point VSF body.
        
        Args:
            vsf_model: The point VSF model.
            contact_params: The contact parameters specifying the friction coefficient and sliding step.
            triangle_mesh: An optional triangle mesh object for visualization.
        """
        self.vsf_model = vsf_model
        self.pose = np.eye(4)
        self.contact_params = contact_params
        self.use_proxy_method = True

        self.perfer: PerfRecorder = DummyRecorder()
        self.verbose = False

        # Setup closest point querier and simulator state
        self.rest_pts_npy = self.vsf_model.rest_points.cpu().numpy()
        from klampt.io import numpy_convert
        from klampt import Geometry3D
        self.klampt_point_cloud = Geometry3D(numpy_convert.from_numpy(self.rest_pts_npy,'PointCloud'))
        self.setup_obj_knn()

        # Reset the simulation state, initialize point contact status
        self.reset()

    @property
    def num_contacts(self):
        """
        Get the number of VSF points in contact
        """    
        return (self.point_object_idx>=0).sum()
        
    def reset(self):
        """
        Reset the simulation state of point VSF.
        
        Clear all contact points and anchor points. 
        """
        # store contact status of points
        self.point_object_idx = np.full(self.vsf_model.rest_points.size(0), -1, dtype=int)
        self.point_local = np.zeros((self.vsf_model.rest_points.size(0), 3))
        self.anchor_local = np.zeros((self.vsf_model.rest_points.size(0), 3))
        self.anchor_normal_local = np.zeros((self.vsf_model.rest_points.size(0), 3))


    def setup_obj_knn(self):
        """
        Setup the nearest neighbor search for the object.
        
        This is used to find the VSF points that are close to the query points.
        """
        self.obj_knn = o3d.core.nns.NearestNeighborSearch(self.rest_pts_npy.astype(np.float32))
        self.obj_knn.hybrid_index()

    def get_proxy_pts(self, query_pts:np.ndarray, prox_margin=1e-2, max_nn_pts=20) -> tuple[np.ndarray, np.ndarray]:
        """
        Find VSF points that are close to the query points.

        NOTE: this only uses the REST points.  Should we use the deformed points?

        :param query_pts: query points in world coordinates, shape (N, 3)
        :param prox_margin: margin for proximity search
        :param max_nn_pts: maximum number of nearest neighbors to search

        :return:
            - prox_idx_ary: indices of the VSF points that are close to the query points
            - prox_pts: VSF points that are close to the query points, in world space 
        """

        query_pts_tsr = o3d.core.Tensor((query_pts-self.pose[:3,3]).dot(self.pose[:3,:3]).astype(np.float32))
        prox_idx_ary = self.obj_knn.hybrid_search(query_pts_tsr, prox_margin, max_nn_pts)[0]
        prox_idx_ary = prox_idx_ary.numpy().reshape(-1)
        prox_idx_ary = prox_idx_ary[prox_idx_ary >= 0]
        
        prox_idx_ary = np.unique(prox_idx_ary)

        if prox_idx_ary.size == 0:
            return np.zeros(0, dtype=int), np.zeros((0, 3))
        else:
            prox_in_contact = self.point_object_idx[prox_idx_ary] >= 0
            prox_idx_ary = prox_idx_ary[~prox_in_contact]
            return prox_idx_ary, self.rest_pts_npy[prox_idx_ary, :].dot(self.pose[:3,:3].T) + self.pose[:3,3]
    
    def slide_contacts(self, state: klamptWorldWrapper, modify_data:bool=False):
        """
        Perform stick/slide logic for the VSF points in contact with the object.
        
        For points that are in contact, check if they are sticking or sliding.
        Stick/slide condition is determined whether the contact force direction is within the friction cone.
        If the point is sticking, do nothing, kinematic update will let the point move with the object.
        If the point is sliding, query the contact point on the object and update the contact point.
        If the contact force direction points outward the surface, remove the contact point.
        
        Args:
            state: the current state of the world
            modify_data: whether to modify the contact data            
        """
        point_idxs = np.where(self.point_object_idx >= 0)[0] 
        if len(point_idxs) == 0:
            return []
        
        self.perfer.start('slide_transform_points')
        rest_pts = transform_points(self.rest_pts_npy[point_idxs],self.pose)
        
        self.perfer.start('slide_transform_points_get_tbody')
        all_Tbody_ary = np.zeros((len(state.name_lst), 4, 4))
        for i, name in enumerate(state.name_lst):
            Tbody = state.bodies_dict[name].getTransform()
            all_Tbody_ary[i] = se3.ndarray(Tbody)
        Tbody_ary = all_Tbody_ary[self.point_object_idx[point_idxs], ...]
        self.perfer.stop('slide_transform_points_get_tbody')
        
        self.perfer.start('slide_transform_points_apply_tbody')
        num_points = self.num_contacts
        assert num_points == len(point_idxs)
        points_homogeneous = np.hstack((self.anchor_local[point_idxs, :], np.ones((num_points, 1))))  # Shape: (N, 4)
        normals_homogeneous = np.hstack((self.anchor_normal_local[point_idxs, :], np.zeros((num_points, 1))))  # Shape: (N, 4)

        # Batch matrix multiplication for points
        transformed_points_homogeneous = np.einsum('nij,nj->ni', Tbody_ary, points_homogeneous)
        world_pts_batch = transformed_points_homogeneous[:, :3]  # Drop the homogeneous coordinate

        # Batch matrix multiplication for normals
        transformed_normals_homogeneous = np.einsum('nij,nj->ni', Tbody_ary, normals_homogeneous)
        world_nms_batch = transformed_normals_homogeneous[:, :3]  # Drop the homogeneous coordinate
        self.perfer.stop('slide_transform_points_apply_tbody')

        world_pts = world_pts_batch
        world_nms = world_nms_batch

        # assert np.allclose(world_pts, world_pts_batch)
        # assert np.allclose(world_nms, world_nms_batch)

        #update the sticking points
        self.point_local[point_idxs] = transform_points(world_pts,np.linalg.inv(self.pose))
        self.perfer.stop('slide_transform_points')

        self.perfer.start('slide_check_stick_condition')
        dr = world_pts - rest_pts
        dr_l2 = np.linalg.norm(dr, axis=1, keepdims=True)
        dr_dot_nm = (dr*world_nms).sum(axis=1, keepdims=True)

        move_bool = (dr_l2>=1e-3).reshape(-1)
        friction_cosin = 1/np.sqrt(1 + self.contact_params.friction_mu**2)
        slide_bool = (dr_dot_nm<friction_cosin*dr_l2).reshape(-1)
        move_and_slide = move_bool & slide_bool & (dr_dot_nm>0).flatten()
        self.perfer.stop('slide_check_stick_condition')

        if self.contact_params.sim_slide and np.any(move_and_slide):
            assert np.all(dr_l2[move_and_slide] >= 1e-9)

            self.perfer.start('slide_compute_slide_direction')
            # NOTE: initially, points have penetration into the surface
            slide_dv = (dr - dr_dot_nm*world_nms)[move_and_slide, :]
            slide_nms = world_nms[move_and_slide, :]
            # check slide_dv * nms == 0
            assert np.allclose((slide_dv*slide_nms).sum(axis=1), 0, atol=1e-6), \
                f"Maximum error: {np.abs((slide_dv*slide_nms).sum(axis=1)).max()}"
            # check slide_dv_l2 > 0
            assert np.all(np.linalg.norm(slide_dv, axis=1) > 1e-9)
            slide_dv /= np.linalg.norm(slide_dv, axis=1, keepdims=True)
            slide_world_pts:np.ndarray = world_pts[move_and_slide, :] - self.contact_params.slide_step*slide_dv
            slide_bodies = self.point_object_idx[point_idxs[move_and_slide]]
            self.perfer.stop('slide_compute_slide_direction')

            # print('slide_world_pts shape:', slide_world_pts.shape)
            
            self.perfer.start('slide_query_contacts')
            new_world_pts = []
            new_world_nms = []
            new_local_pts = []
            new_local_nms = []
            import klampt
            for j, pt in enumerate(slide_world_pts):
                sdf = state.local_sdf_lst[slide_bodies[j]]   # type: klampt.Geometry3D
                Tbody = state.bodies_dict[state.name_lst[slide_bodies[j]]].getTransform()
                sdf.setCurrentTransform(*state.bodies_dict[state.name_lst[slide_bodies[j]]].getTransform())
                res = sdf.distance_point(pt)
                assert res.hasClosestPoints
                new_world_pts.append(list(res.cp1))
                new_world_nms.append(list(res.grad2))
                new_local_pts.append(se3.apply(se3.inv(Tbody), res.cp1))
                new_local_nms.append(so3.apply(so3.inv(Tbody[0]), res.grad2))
            self.perfer.stop('slide_query_contacts')

            self.perfer.start('slide_update_contacts')
            world_pts[move_and_slide] = np.array(new_world_pts)
            world_nms[move_and_slide] = np.array(new_world_nms)

            #update the points that have moved
            self.anchor_local[point_idxs[move_and_slide]] = np.array(new_local_pts)
            self.anchor_normal_local[point_idxs[move_and_slide]] = np.array(new_local_nms)
            self.point_local[point_idxs[move_and_slide]] = transform_points(np.array(new_world_pts),np.linalg.inv(self.pose))
        
            if self.verbose and np.sum(move_and_slide) > 0:
                raise NotImplementedError("Visual debugging not implemented in new API")
                from ..visualize.o3d_visualization import create_motion_lines
                arm_pcd = self.klampt_world_wrapper.get_all_pcd()
                [pcd.paint_uniform_color([0.5, 0.5, 0.5]) for pcd in arm_pcd]
                pcd1, pcd2, motion_lines = create_motion_lines(self.rest_pts_npy[point_idxs[move_and_slide]], 
                                                                world_pts[move_and_slide], return_pcd=True)
                
                slide_pcd = o3d.geometry.PointCloud()
                slide_pcd.points = o3d.utility.Vector3dVector(world_pts[move_and_slide])
                # slide_pcd.normals = o3d.utility.Vector3dVector(-slide_dv)
                slide_pcd.normals = o3d.utility.Vector3dVector(world_nms[move_and_slide])
                o3d.visualization.draw_geometries(arm_pcd + [pcd1, slide_pcd, motion_lines], window_name='Slide contacts')

            # recompute points displacements
            dr = world_pts - self.rest_pts_npy[point_idxs]
            dr_l2 = np.linalg.norm(dr, axis=1)
            dr_dot_nm = (dr*world_nms).sum(axis=1)
            self.perfer.stop('slide_update_contacts')
            

        move_bool = (dr_l2>=1e-3).reshape(-1)
        neg_dot_bool = (dr_dot_nm<0.1*dr_l2).reshape(-1)
        if self.verbose:
            print('move_bool:', np.sum(move_bool))
            print('neg_dot_bool:', np.sum(neg_dot_bool))

        remove_points = np.where(np.logical_and(neg_dot_bool, move_bool))[0]
        if modify_data:
            self.point_object_idx[point_idxs[remove_points]] = -1
        return remove_points

    def step(self, state : klamptWorldWrapper, dt:float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,torch.Tensor]:
        """
        Steps the simulation.  Returns the set of contact indices, bodies, points, and forces
        
        The step function in point VSF performs the following steps:
        1. detect stick/slide contacts, this step will keep sticking points, slide points
        on the contact surface, and remove points whose contact force points outward the surface.
        2. get new set of points in contact with the object, using query_point_contacts function, 
        when using proxy method, the query points are the VSF points that are close to the object.
        3. compute the force tensor for the points in contact, using the VSF model.
        
        Args:
            state: the current state of the world
            dt: the time step, the number is not used since running quasistatic simulation
        
        TODO: convert the return values to a dataclass
        
        Returns:
            The set of contact indices, bodies, points, and forces
            indices: the indices of the VSF points in contact
            bodies: the indices of the objects that the VSF points are in contact with
            points: the contact points in world coordinates
            forces: the contact forces in world coordinates
        """
        # Remove detached contacts
        self.perfer.start('slide')
        self.slide_contacts(state, modify_data=True)
        self.perfer.stop('slide')

        if self.use_proxy_method:
            #use proxy method to find contacts
            self.perfer.start('prox')
            all_pcd = sum(state.get_all_pcd(format='open3d'), o3d.geometry.PointCloud())
            query_points = np.array(all_pcd.points)
            prox_idx, prox_pts = self.get_proxy_pts(query_points)
            self.perfer.stop('prox')
            
            if prox_idx.size == 0:
                return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 3)), torch.zeros((0, 3),dtype=self.vsf_model.rest_points.dtype,device=self.vsf_model.rest_points.device)
        
            self.perfer.start('contact')
            pc_obj_idxs, depths, contact_pts, contact_nms = state.query_point_contacts(prox_pts, padding1=self.contact_params.geometry_padding1,padding2=self.contact_params.geometry_padding2)
        else:
            #just do collision detection with whole point cloud
            self.perfer.start('contact')
            self.klampt_point_cloud.setCurrentTransform(*se3.from_ndarray(self.pose)) 
            pc_obj_idxs, depths, contact_pts, contact_nms = state.query_point_contacts(self.klampt_point_cloud, padding1=self.contact_params.geometry_padding1,padding2=self.contact_params.geometry_padding2)
            prox_idx = np.arange(self.rest_pts_npy.size(0))

        # update contact status for newly added points
        for i in range(pc_obj_idxs.shape[0]):
            if pc_obj_idxs[i] < 0:
                continue
            body = state.bodies_dict[state.name_lst[pc_obj_idxs[i]]]
            body_transform_inv = se3.inv(body.getTransform())
            pt_idx = prox_idx[i]
            if self.point_object_idx[pt_idx] >= 0:  #do logic to determine whether to switch to a new anchor point
                d = np.linalg.norm(se3.apply(se3.from_ndarray(self.pose),self.point_local[pt_idx]) - self.rest_pts_npy[pt_idx])
                if depths[i] < d + 1e-3:  #new contact is too shallow, keep existing one
                    continue
            self.point_object_idx[pt_idx] = pc_obj_idxs[i]
            self.point_local[pt_idx] = se3.apply(se3.inv(se3.from_ndarray(self.pose)), contact_pts[i])
            self.anchor_local[pt_idx] = se3.apply(body_transform_inv, contact_pts[i])
            self.anchor_normal_local[pt_idx] = so3.apply(body_transform_inv[0], contact_nms[i])
        self.perfer.stop('contact')

        #compute force tensor
        # TODO: put the following as reusable function in the VSF model?
        contact_indices = np.where(self.point_object_idx >= 0)[0]
        bodies = self.point_object_idx[contact_indices]
        vsf_pts = self.point_local[contact_indices]
        assert isinstance(contact_indices, np.ndarray) and len(contact_indices.shape)==1
        assert isinstance(vsf_pts, np.ndarray) and len(vsf_pts.shape)==2
        vsf_forces = self.vsf_model.compute_forces(torch.tensor(contact_indices,dtype=torch.long,device=self.vsf_model.rest_points.device),
                                                   torch.from_numpy(vsf_pts).to(self.vsf_model.rest_points.device))
        world_pts = transform_points(vsf_pts,self.pose)
        world_forces = transform_directions(vsf_forces,self.pose)
        return contact_indices,bodies,world_pts,world_forces
        
    def state(self):
        """
        Get the current state of the point VSF body
        
        TODO: converts the state to a dataclass
        
        Returns:
            A dictionary containing the pose, point object index, point local, 
            anchor local, and anchor normal local
        """
        return {
            'pose': self.pose,
            'point_object_idx': self.point_object_idx,
            'point_local': self.point_local,
            'anchor_local': self.anchor_local,
            'anchor_normal_local': self.anchor_normal_local
        } 
    
    def load_state(self, state: dict):
        """
        Load the simulation state of the point VSF body from a dictionary
        
        Args:
            state: the dictionary containing the pose, point object index, 
            point local, anchor local, and anchor normal local
        """
        self.pose = state['pose']
        self.point_object_idx = state['point_object_idx']
        self.point_local = state['point_local']
        self.anchor_local = state['anchor_local']
        self.anchor_normal_local = state['anchor_normal_local']

    def deformed_points(self, world=True) -> np.ndarray:
        """Returns all the points of the VSF, deformed by the current state"""
        res = self.rest_pts_npy.copy()
        res[self.point_object_idx>=0] = self.point_local[self.point_object_idx>=0]
        if world:
            return transform_points(res,self.pose)
        return res

    def deformed_points_torch(self, world=True) -> torch.Tensor:
        """Returns all the points of the VSF, deformed by the current state"""
        res = self.vsf_model.rest_points.clone()
        res[self.point_object_idx>=0] = torch.tensor(self.point_local[self.point_object_idx>=0],dtype=res.dtype,device=res.device)
        if world:
            return transform_points(res,self.pose)
        return res
    
    def get_obj_pcd(self, color_method:str='contact', dtype='o3d'):
        """
        Get the object point cloud with color based on contact status
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.rest_points_npy)

        colors = np.zeros_like(self.rest_points_npy)
        in_contact = self.point_object_idx >= 0
        if color_method == 'contact':
            colors[in_contact, :] = [0.0, 1.0, 0.0]
            colors[~in_contact, :] = [1.0, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if dtype == 'o3d':
            return pcd
        elif dtype == 'klampt':
            from klampt.io import open3d_convert
            return open3d_convert.from_open3d(pcd)

    def get_deform_motion_lines(self, slide_color=[0.0, 1.0, 0.0], world=False, 
                                verbose=False) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.LineSet]:
        """
        Returns the motion lines of the deformed points
        
        Args:
            slide_color: the color of the sliding points
            world: whether to return the motion lines in world coordinates
            verbose: whether to print debug information
        
        Returns:
            rest_pcd: the rest point cloud
            curr_pcd: the current point cloud
            motion_lines: the motion lines
        """
        contact_obj_idx = np.where(self.point_object_idx >= 0)[0]
        if len(contact_obj_idx) == 0:
            return [], [], []
        rest_pts = self.rest_pts_npy[self.point_object_idx>=0, :]
        curr_pts = self.point_local[self.point_object_idx>=0, :]
        curr_nms = self.anchor_normal_local[self.point_object_idx>=0, :]
        
        if world:
            rest_pts = transform_points(rest_pts, self.pose)
            curr_pts = transform_points(curr_pts, self.pose)

        # TODO: color returned geometry based on slide status
        remove_obj_idx = np.concatenate(self.slide_contacts(modify_data=False, verbose=verbose))
        remove_bool_ary = np.isin(contact_obj_idx, remove_obj_idx)
        if verbose:
            print('remove bool ary shape:', remove_bool_ary.shape)
            print('number slides:', np.sum(remove_bool_ary))

        from ..visualize.o3d_visualization import create_motion_lines
        rest_pcd, curr_pcd, motion_lines = create_motion_lines(rest_pts, curr_pts, return_pcd=True)
        curr_pcd.normals = o3d.utility.Vector3dVector(curr_nms)

        colors = np.array(curr_pcd.colors)
        colors[remove_bool_ary, :] = slide_color
        curr_pcd.colors = o3d.utility.Vector3dVector(colors)

        return rest_pcd, curr_pcd, motion_lines

    @property
    def device(self):
        """Get the device of the VSF model"""
        return self.vsf_model.device
    
    @property
    def dtype(self):
        """Get the dtype of the VSF model"""
        return self.vsf_model.dtype