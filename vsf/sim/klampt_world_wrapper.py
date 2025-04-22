import klampt
from klampt import WorldModel,RigidObjectModel, RobotModelLink
from klampt.model.typing import RigidTransform
from klampt.model.typing import Rotation, Vector3
from klampt.math import se3, so3
import numpy as np
import open3d as o3d
import os, sys
from dataclasses import dataclass
from copy import deepcopy
from typing import Union

@dataclass
class AttachmentInfo:
    """
    Stores information about an attachment between two bodies.
    """
    object_model_name: str  # Name of the object model being attached
    parent_model_name: str  # Name of the parent model to which the object is attached
    relative_transformation: np.ndarray  # 4x4 homogeneous transformation matrix

    def __post_init__(self):
        """
        Ensures that the relative_transformation is a 4x4 numpy array.
        """
        if not isinstance(self.relative_transformation, np.ndarray):
            raise TypeError("relative_transformation must be a numpy ndarray.")
        if self.relative_transformation.shape != (4, 4):
            raise ValueError("relative_transformation must be a 4x4 matrix.")



class klamptWorldWrapper:
    """
    A wrapper for the Klamp't world.

    Stores list of bodies in a flat dict, and gives an ordering to the names
    of objects in the world.  PCDs and SDFs are stored with each geometry so
    that you can query contact points more quickly.
    """

    def __init__(self) -> None:
        """
        Initialize the Klamp't world
        """
        self.world = WorldModel()

        self.control_type_dict = {}
        self.attachments:list[AttachmentInfo] = []

        # provide an ordered list of object names
        self.name_lst:list[str] = []
        self.bodies_dict:dict[str, Union[RigidObjectModel, RobotModelLink]] = {}
        self.local_pcd_lst:list[o3d.geometry.PointCloud] = []
        self.local_sdf_lst:list[klampt.Geometry3D] = []

    @staticmethod
    def from_world(world: WorldModel) -> 'klamptWorldWrapper':
        """
        Create a Klamp't world wrapper from an existing Klamp't world
        """
        world_wrapper = klamptWorldWrapper()
        world_wrapper.world = world
        for i in range(world.numRigidObjects()):
            rigid_object = world.rigidObject(i)
            name = rigid_object.getName()
            world_wrapper.name_lst.append(name)
            world_wrapper.bodies_dict[name] = rigid_object
            world_wrapper.control_type_dict[name] = 'rigid'
        for i in range(world.numRobots()):
            robot = world.robot(i)
            name = robot.getName()
            for j in range(robot.numLinks()):
                link = robot.link(j)
                assert link.name not in world_wrapper.bodies_dict,"Can't have duplicate body names in klamptWorldWrapper"
                world_wrapper.name_lst.append(link.name)
                world_wrapper.bodies_dict[link.name] = link
            world_wrapper.control_type_dict[name] = 'robot'
        return world_wrapper

    @property
    def num_bodies(self):
        return len(self.bodies_dict)
    

    def add_robot(self, name: str, robot_file_name: str):
        """
        Add a robot to the klamp't world
        
        :param name: name of the robot
        :param robot_file_name: URDF file name of the robot
        """
        # Assert the robot file exists
        assert os.path.exists(robot_file_name), "Robot file {} does not exist".format(robot_file_name)
    
        robot = self.world.loadRobot(robot_file_name)
        
        robot.setName(name)
        self.control_type_dict[name] = 'robot'        
        
        for link_idx in range(robot.numLinks()):
            link_name = robot.link(link_idx).name
            if link_name in self.bodies_dict:
                raise ValueError("Duplicate link name")
            self.name_lst.append(link_name)
            self.bodies_dict[link_name] = robot.link(link_idx)

    def add_geometry(self, name: str, geom: klampt.Geometry3D, geom_type: str, 
                     parent_name: str=None, parent_relative_transform: np.ndarray=None):
        """
        Add a geometry to the klamp't world
        
        :param name: name of the geometry
        :param geom: geometry object
        :param geom_type: type of the geometry, "rigid" or "deformable"
        :param parent_name: name of the parent geometry
        """        
        # Add a rigid object to the world
        rigid_object = self.world.makeRigidObject(name)
        self.control_type_dict[name] = geom_type
        self.name_lst.append(name)
        self.bodies_dict[name] = rigid_object
        rigid_object.geometry().set(geom)
        
        if parent_name is not None:
            if parent_name not in self.name_lst:
                raise ValueError("Parent object does not exist")
            assert parent_relative_transform is not None, "Parent relative transform must be provided"
            self.attachments.append(AttachmentInfo(name, parent_name, parent_relative_transform))
        self.update_attachments()

    def add_geometry_from_file(self, name: str, geom_file_name: str, geom_type: str,
                               parent_name: str=None, parent_relative_transform: np.ndarray=None):
        """
        Add a geometry to the klamp't world
        
        :param name: name of the geometry
        :param geom_file_name: file name of the geometry
        :param geom_type: type of the geometry, "rigid" or "deformable"
        :param parent_name: name of the parent geometry
        :param parent_relative_transform: relative transformation between the geometry and its parent
        """

        # Load the geometry from the file
        if geom_file_name.endswith('.vtk'):
            # NOTE: use .vtk file to load Punyo mesh
            from ..utils.vtk_utils import unpack_vtk_mesh
            vertices, triangles, _, _ = unpack_vtk_mesh(geom_file_name)
            triangle_mesh = klampt.TriangleMesh()
            triangle_mesh.setVertices(vertices)
            triangle_mesh.setIndices(triangles)
            geom = klampt.Geometry3D()
            geom.setTriangleMesh(triangle_mesh)
        else:
            # geom = klampt.Geometry3D()
            # geom.loadFile(geom_file_name)
            #this function should be used instead of native Klampt loaders due to a known Assimp configuration issue
            from ..utils.klampt_utils import load_trimesh_preserve_vertices
            geom = load_trimesh_preserve_vertices(geom_file_name)

        self.add_geometry(name, geom, geom_type, parent_name, parent_relative_transform)
    
    def apply_control(self, name:str, control: np.ndarray):
        """
        Apply control to an object in the world

        :param name: name of the object
        :param control: control array for the object
        """
        if self.control_type_dict[name] == 'robot':
            self.update_robot(name, control)
        elif self.control_type_dict[name] == 'rigid':
            rotation, position = se3.from_ndarray(control)
            self.update_geometry(name, rotation, position)
        elif self.control_type_dict[name] == 'deformable':
            self.update_deformable(name, control)
    
    def update_robot(self, name: str, robot_drivers: np.ndarray):
        """
        Update the configuration of a robot in the Klamp't world

        :param name: name of the robot
        :param robot_config: new configuration of the robot
        """
        assert self.control_type_dict[name] == 'robot', \
            "Only apply update robot configuration on robot model."
        robot = self.world.robot(name)
        
        assert len(robot_drivers) == robot.numDrivers(),"The robot control must have length #drivers"
        config = robot.configFromDrivers(robot_drivers.tolist())
        robot.setConfig(config)
        self.update_attachments()
        
    def update_attachments(self):
        """
        Automatically update objects attached to kinematically controlled objects.
        This function should only be called internally.
        
        NOTE: this is a workaround for the lack of proper attachment support in Klamp't,
        ideally should be substituted by automatic kinematic update in Klamp't.        
        """
        for attachment in self.attachments:
            parent_model = self.bodies_dict[attachment.parent_model_name]
            child_model = self.bodies_dict[attachment.object_model_name]
            assert isinstance(child_model, RigidObjectModel), \
                "Child model must be a rigid object, cannot mount robot link on another robot"

            parent_R, parent_t = parent_model.getTransform()
            relative_transformation = se3.from_ndarray(attachment.relative_transformation)
            child_R, child_t = se3.mul((parent_R, parent_t), relative_transformation)
            child_model.setTransform(child_R, child_t)
    
    def update_geometry(self, name: str, rotation: Rotation, position: Vector3):
        """
        Update the rigid transformation of a geometry

        :param name: name of the geometry
        :param rotation: new rotation, should be in Klampt Rotation format  
        :param position: new translation, should be in Klampt Vector3 format
        """
        rigid_model:RigidObjectModel = self.world.rigidObject(name)
        R, t = rotation, position
        rigid_model.setTransform(R, t)

    def update_deformable(self, name: str, control: np.ndarray):
        """
        This function update the triangle meshes of a deformable object in the Klamp't world
        
        TODO: Currently only update the triangle mesh, should trigger SDF/point cloud update in the future.
        
        :param name: name of the deformable object
        :param control: control should be Nx3 vertex positions
        """
        geom = self.world.rigidObject(name).geometry()
        geom.getTriangleMesh().setVertices(control)
    
    def convert_geometry(self, geom_idx_lst: list[int], src_type: str, tgt_type: str):
        """
        Convert the geometry type of the rigid objects in the Klamp't world

        :param geom_idx_lst: list of geometry indices
        :param src_type: source geometry type
        :param tgt_type: target geometry type

        "src_type" and "tgt_type" can be one of the following:
        - 'VolumeGrid'
        - 'TriangleMesh'
        - 'PointCloud'
        Note this function will not change geometry not matching the "src_type"
        """
        for geom_idx in geom_idx_lst:
            rigid_model = self.world.rigidObject(geom_idx)

            if rigid_model.geometry().type() == src_type:
                geom = rigid_model.geometry()

                new_geom = geom.convert(tgt_type).copy()
                if tgt_type == 'VolumeGrid':
                    rigid_model.geometry().setVolumeGrid(new_geom.getVolumeGrid())
                elif tgt_type == 'TriangleMesh':
                    rigid_model.geometry().setTriangleMesh(new_geom.getTriangleMesh())
                elif tgt_type == 'PointCloud':
                    rigid_model.geometry().setPointCloud(new_geom.getPointCloud())
    
    def get_transform(self, model:Union[RigidObjectModel,RobotModelLink], 
                      mode='local2world', format='klampt'):
        """
        Get the rigid transformation of a geometry in the Klamp't world
        Here the model can be a rigid object or a robot link, 
        we can get transformation from world to local or local to world.
        The return type can be Klamp't SE3 transformation or 4x4 homogeneous 
        numpy transformation matrix.
        
        :param model: the model of the geometry
        :param mode: transformation mode, 'local2world' or 'world2local'
        :param format: transformation format, 'klampt' or 'numpy'
        
        :return: the transformation matrix, in user specified format
        """
        H = model.getTransform()
        
        # Default mode is local to world
        if mode == 'world2local':
            H = se3.inv(H)

        if format == 'klampt':
            return H
        elif format == 'numpy':
            return se3.ndarray(H)
    
    def get_geom_trans_dict(self, mode='local2world', format='klampt') -> dict[str, np.ndarray]:
        """
        Get the transformation of all geometries in the Klamp't world
        """
        geom_trans_dict = { key: self.get_transform(body, mode, format) 
                            for key, body in self.bodies_dict.items() }        
        return geom_trans_dict
    
    def setup_local_sdf_lst(self, resolution=0.01):
        """
        Setup the the signed distance field list of all geometries in the Klamp't world.
        All SDFs are in the geometry local frame, saved in the `self.local_sdf_lst` list.
        
        NOTE: this function should be substituted by multi-geometry support in Klamp't,
        ideally should be saved in a structure accssible in the `RigidObjectModel` or 
        `RobotModelLink` object.
        
        :param resolution: resolution of the signed distance field
        """
        self.local_sdf_lst = []
        for name in self.name_lst:
            model = self.bodies_dict[name]
            if model.geometry().type() == 'TriangleMesh':
                geom = model.geometry().convert('VolumeGrid', param=resolution)
            else:
                geom = None
                # raise ValueError("Can only convert triangle mesh to signed distance field")
            self.local_sdf_lst.append(geom)

    def setup_local_pcd_lst(self, sample_backend='klampt', dispersion=0.01, num_of_pts=1000):
        """
        Setup the open3d point cloud list, save in the `self.local_pcd_lst` list.
        The point clouds are used for proximity queries to detect contact points with 
        PointVSF. 
        
        NOTE: the dispersion in the point cloud sampling should match the 
        resolution of PointVSF to ensure accurate contact detection.
        Too coarse point cloud may miss contacts, while too fine point cloud
        may slow down the simulation.
        
        TODO: this function should be substituted by multi-geometry support in Klamp't,
        ideally should be saved in a structure accssible in the `RigidObjectModel` or
        `RobotModelLink` object.
        
        :param sample_backend: the backend to sample the point cloud, 'klampt' or 'open3d'
        :param dispersion: dispersion of the point cloud, only used when sample_backend is 'klampt'
        :param num_of_pts: number of points in the point cloud, only used when sample_backend is 'open3d'
        """
        from klampt.io import open3d_convert
        self.local_pcd_lst = []
        for name in self.name_lst:
            model = self.bodies_dict[name]
            if model.geometry().type() == 'TriangleMesh':
                geom = model.geometry()
                if sample_backend == 'klampt':
                    klampt_pcd = geom.convert('PointCloud', param=dispersion)
                    pcd = open3d_convert.to_open3d(klampt_pcd.getPointCloud())
                elif sample_backend == 'open3d':
                    o3d_mesh = open3d_convert.to_open3d(geom.getTriangleMesh())
                    pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_of_pts)
                    # o3d.visualization.draw_geometries([pcd])
            else:
                pcd = o3d.geometry.PointCloud()
                print(f"Warning: model {name} type {model.geometry().type()} is not triangle mesh")
                # raise ValueError("Can only convert triangle mesh to point cloud: ", 
                #                  geom_idx, geom.geometry().type())
            self.local_pcd_lst.append(pcd)

    def get_all_pcd(self, format='open3d'):
        """
        Get all point clouds of the objects in the Klamp't world.
        All point clouds are transformed in the world coordinate frame.
        
        :param format: format of the point cloud, 'open3d' or 'numpy'
        """
        assert len(self.local_pcd_lst) == len(self.name_lst), \
            "Please setup the local point cloud list first"
        
        transform_dict = self.get_geom_trans_dict(mode='local2world', format='numpy')
        transform_lst = [transform_dict[name] for name in self.name_lst]
        # assert len(transform_lst) == len(self.local_pcd_lst), "Transform list length mismatch"

        world_pcd_lst = [deepcopy(pcd) for pcd in self.local_pcd_lst]
        for geom_idx in range(len(world_pcd_lst)):
            if world_pcd_lst[geom_idx] is None:
                continue
            world_pcd_lst[geom_idx].transform(transform_lst[geom_idx])
        
        return world_pcd_lst
    
    def find_closest_point(self, query_points: np.ndarray, geom_idx : int=None, 
                           verbose: bool=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the closest point from a set of query points to a geometry in
        the Klamp't world.  The geometry is given by index `geom_idx`.
        
        TODO: currently this function only supports points query in a for loop, 
        should be substituted by batch query in the future.
        
        Inputs:
        :param query_points: a Nx3 numpy array of query points
        :param geom_idx: index of the geometry in the Klamp't world
        :param verbose: whether to visualize the query results
        
        Outputs:
        :return: a tuple of closest points and normals
        """        
        name = self.name_lst[geom_idx]
        model = self.bodies_dict[name]
        R, t = model.getTransform()
        H = se3.ndarray((R, t))
        
        sdf:klampt.Geometry3D = self.local_sdf_lst[geom_idx]
        sdf.setCurrentTransform(R, t)
        
        closest_pts = []
        closest_nms = []
        for query_point in query_points:
            query_result = sdf.distance_point(query_point)
            
            # print("query_point:", query_point)
            # print("closest_point:", np.array(query_result.cp1))
            # print("closest_normal:", np.array(query_result.grad1))
            
            closest_pts.append(np.array(query_result.cp1))
            closest_nms.append(np.array(query_result.grad1))
                        
            # print("closest_pts shape:", closest_pts[-1].shape)
            # print("closest_nms shape:", closest_nms[-1].shape)
            
        closest_pts = np.stack(closest_pts).reshape(-1, 3)
        closest_nms = -np.stack(closest_nms).reshape(-1, 3)
        # print("closest_pts shape:", closest_pts.shape)
        # print("closest_nms shape:", closest_nms.shape)
        
        if verbose:
            world_pcd = deepcopy(self.local_pcd_lst[geom_idx])
            world_pcd.transform(H)
            world_pcd.paint_uniform_color([0.1, 1.0, 0.1])

            o3d_query_pcd = o3d.geometry.PointCloud()
            o3d_query_pcd.points = o3d.utility.Vector3dVector(query_points)
            o3d_query_pcd.paint_uniform_color([0.7, 0.7, 0.7])
            
            closest_pcd = o3d.geometry.PointCloud()
            closest_pcd.points = o3d.utility.Vector3dVector(closest_pts)
            closest_pcd.normals = o3d.utility.Vector3dVector(closest_nms)
            closest_pcd.paint_uniform_color([1.0, 0.1, 0.1])
            
            o3d.visualization.draw_geometries([world_pcd, closest_pcd, o3d_query_pcd], 
                                              window_name='reproject points')
        
        return closest_pts, closest_nms

    def query_point_contacts(self, query_points: Union[np.ndarray,klampt.Geometry3D], padding1: float=1e-3, 
                             padding2: float=1e-3, body_idx_lst=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Query whether the points are in contact with any objects in the Klamp't world.

        This function uses the `geometry.contacts` function to query the contact points,
        which applies the boundary layer method for collision checking.

        TODO: The collision query result should be saved in a specific dataclass
        to make the query results more readable.

        Args:
            query_points (np.ndarray | klampt.Geometry3D): 
                A (N,3) NumPy array of query points.
            padding1 (float, optional): 
                Padding of the query points. Default is `1e-3`.
            padding2 (float, optional): 
                Padding of the geometry. Default is `1e-3`.
            body_idx_lst (list, optional): 
                List of body indices to query. If `None`, all bodies are queried.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:

            - **body_idx** (`np.ndarray`): Indices of bodies in contact, or `-1` if no contact.
            - **depths** (`np.ndarray`): Penetration depths.
            - **contact_pts** (`np.ndarray`): Closest points on body surfaces (in world coordinates).
            - **contact_nms** (`np.ndarray`): Contact normals (in world coordinates).

            The length of these arrays is the same as the number of query points.
        """
        if body_idx_lst is None:
            body_idx_lst = list(range(len(self.name_lst)))
        
        # First convert points to Klampt point cloud, for speed
        if not isinstance(query_points, klampt.Geometry3D):
            from klampt.io import numpy_convert
            prox_pc = numpy_convert.from_numpy(query_points, type='PointCloud')
            prox_geom = klampt.Geometry3D()
            prox_geom.setPointCloud(prox_pc)
        else:
            prox_geom = query_points
            query_points = prox_geom.getPointCloud().points

        # Record which points have established contact
        body_idx = np.full(query_points.shape[0], -1, dtype=int)
        depths = np.zeros(query_points.shape[0])
        contact_pts = np.zeros((query_points.shape[0], 3))
        contact_nms = np.zeros((query_points.shape[0], 3))

        for idx in body_idx_lst:
            name = self.name_lst[idx]
            model = self.bodies_dict[name]
            R, t = model.getTransform()

            if self.local_sdf_lst[idx] is None:  # no SDF defined
                continue

            sdf = self.local_sdf_lst[idx]
            assert isinstance(sdf, klampt.Geometry3D)            
            sdf.setCurrentTransform(R, t)
            
            # Volume grid contacts point cloud
            contact_ret = sdf.contacts(prox_geom, padding1, padding2)

            if len(contact_ret.elems2) == 0:
                # no contact
                continue

            for c in range(len(contact_ret.elems1)):
                pt_idx = contact_ret.elems2[c]
                d = contact_ret.depths[c]
                if d > depths[pt_idx]:
                    depths[pt_idx] = d
                    body_idx[pt_idx] = idx
                    contact_pts[pt_idx] = np.array(contact_ret.points1[c])
                    contact_nms[pt_idx] = np.array(contact_ret.normals[c])
        return body_idx, depths, contact_pts, contact_nms
        
    def get_geometry(self, name: str) -> klampt.Geometry3D:
        """
        Get the geometry of an object in the Klamp't world
        
        Inputs:
        :param name: name of the object
        
        Outputs:
        :return: the geometry of the object
        """
        return self.bodies_dict[name].geometry()
    
