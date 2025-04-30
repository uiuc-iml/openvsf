from .core.neural_vsf import NeuralVSF,NeuralVSFConfig
from .core.point_vsf import PointVSF,PointVSFConfig
from .core.vsf_factory import VSFFactory, VSFFactoryConfig, ViewConfig, VSFRGBDCameraFactory, VSFRGBDCameraFactoryConfig
from .estimator.neural_vsf_estimator import NeuralVSFEstimatorConfig
from .utils.data_utils import sdf_to_points
import torch
import os
import klampt
from klampt.math import se3
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Union, Tuple, Optional

@dataclass
class AnyVSFConfig:
    """Describes how to load or create a VSF via any built-in method.

    Attributes:
        type (str): The type of VSF to create.  Allowed values are:
            - 'neural': Neural VSF
            - 'point': Point-based VSF
        init_method (str): The initialization method.  Allowed values are:
            - 'file': Load VSF from a file or folder.
            - 'config': Create an empty VSF from point_vsf_config / neural_vsf_config
            - 'mesh': Create an empty VSF from a triangle mesh named in path.
            - 'factory': Create an empty VSF from a point cloud and the factory_config
            - 'rgbd': Create an empty VSF from RGBD image and the rgbd_factory_config.
        path (str): The path to the file, folder, or mesh.
        point_vsf_config (Optional[PointVSFConfig]): The point-based VSF configuration
            if init_method is 'config'.
        neural_vsf_config (Optional[NeuralVSFConfig]): The neural VSF configuration
            if init_method is 'config'.
        factory_config (Optional[VSFFactoryConfig]): The factory configuration if
            init_method is 'factory'.
        rgbd_factory_config (Optional[VSFRGBDCameraFactoryConfig]): The RGBD factory
            configuration if init_method is 'rgbd'.
    """
    type : str = ''
    init_method : str = ''
    path : str = ''
    point_vsf_config : Optional[PointVSFConfig] = None
    neural_vsf_config : Optional[NeuralVSFConfig] = None
    factory_config : Optional[VSFFactoryConfig] = None
    rgbd_factory_config: Optional[VSFRGBDCameraFactoryConfig] = None
    

def vsf_from_file(file : str) -> Union[NeuralVSF,PointVSF]:
    """Loads a VSF from a file or folder."""
    if os.path.isdir(file):
        if os.path.exists(os.path.join(file,'points.npy')) and os.path.exists(os.path.join(file,'K.npy')):
            res = PointVSF(np.zeros((0,3)))
            res.load(file)
            return res
        else:
            raise ValueError("Directory "+file+" does not contain a valid PointVSF")
    elif file.endswith('.npz'):
        res = PointVSF(np.zeros((0,3)))
        res.load(file)
        return res
    else:
        config = NeuralVSFConfig(([0,0,0],[1,1,1]))
        res = NeuralVSF(config)
        res.load(file)
        res.config.aabb = res.vsfNetwork.aabb
        return res

def vsf_from_points(points : np.ndarray) -> PointVSF:
    """Creates an empty point-based VSF from a set of points."""
    return PointVSF(points)

def vsf_from_box(bmin : np.ndarray, bmax : np.ndarray, type = 'point',
                 shape : Tuple[int,int,int] = None,
                 resolution : float = None) -> PointVSF:
    """Creates an empty point-based or neural VSF from a box.
    
    If type is 'point', the VSF is point-based.  Either `shape` or
    resolution must be given.  If `shape` is given, it gives
    the number of points in each dimension.  If `resolution` is given,
    it gives the resolution of the voxels used to create the point
    grid.

    If type is 'neural', the VSF is neural. 
    """
    if type == 'point':
        if shape is None:
            assert resolution is not None,"Either shape or resolution must be provided"
            shape = [int(np.ceil((bmax[i]-bmin[i])/resolution)) for i in range(3)]
        rest_pts = torch.meshgrid(*[torch.linspace(bmin[i],bmax[i],shape[i]) for i in range(3)],indexing='ij')
        rest_pts = [dim.reshape(-1) for dim in rest_pts]
        rest_pts = torch.stack(rest_pts,dim=1)
        assert rest_pts.shape[1] == 3
        return PointVSF(rest_pts)
    else:
        config = NeuralVSFConfig(aabb = (bmin.tolist(),bmax.tolist()))
        return NeuralVSF(config)

def vsf_from_config(config : Union[NeuralVSFConfig,PointVSFConfig,AnyVSFConfig]) -> Union[NeuralVSF,PointVSF]:
    """Returns an empty vsf from a config."""
    if isinstance(config,AnyVSFConfig):
        if config.init_method in ['','file']:
            return vsf_from_file(config.path)
        elif config.init_method == 'config':
            if config.type == 'point':
                return vsf_from_config(config.point_vsf_config)
            else:
                return vsf_from_config(config.neural_vsf_config)
        elif config.init_method == 'mesh':
            return vsf_from_mesh(config.path, vsf_type=config.type)
        elif config.init_method == 'factory':
            factory = VSFFactory(config.factory_config)
            geom = klampt.Geometry3D()
            if not geom.loadFile(config.path):
                raise RuntimeError("Unable to load geometry file "+config.path)
            return factory.process(geom)
        elif config.init_method == 'rgbd':
            if config.type == 'point':
                factory = VSFRGBDCameraFactory(config.rgbd_factory_config)
                pcd = o3d.io.read_point_cloud(config.path)
                return factory.process(pcd)
            else:
                import imageio
                import json
                rgb_image = imageio.imread(os.path.join(config.path, "color_img.jpg"))
                depth_image = imageio.imread(os.path.join(config.path, "depth_img.png"))
                depth_scale = 1000.0
                depth_trunc = 2.0

                intrinsic = json.load(open(os.path.join(config.path, "intrinsic.json")))
                intrinsic = np.array([[intrinsic['fx'], 0, intrinsic['cx']],
                                    [0, intrinsic['fy'], intrinsic['cy']],
                                    [0, 0, 1]])
                extrinsic = json.load(open(os.path.join(config.path, "extrinsic.json")))
                extrinsic = np.array(extrinsic['cam2world'])
                bmin, bmax = np.load(os.path.join(config.path, "aabb.npy"))

                return vsf_from_rgbd(rgb_image, depth_image, bmin, bmax, intrinsic, extrinsic, depth_scale=depth_scale, depth_trunc=depth_trunc, type='neural')
        else:
            raise ValueError("Invalid init_method "+config.init_method)
    elif isinstance(config,PointVSFConfig):
        if config.rest_points is not None:
            return PointVSF(config.rest_points, axis_mode=config.axis_mode, features=config.features)
        else:
            return PointVSF(rest_points=None, bbox=config.bbox, voxel_size=config.voxel_size, axis_mode=config.axis_mode, features=config.features)
    else:
        assert isinstance(config,NeuralVSFConfig)
        return NeuralVSF(config)

def vsf_from_mesh(mesh : Union[str,klampt.TriangleMesh], 
                  grid_size=128, vsf_type='neural', sdf_thres=0.01) -> Union[NeuralVSF, PointVSF]:
    """Creates an empty neural VSF from a mesh."""
    if isinstance(mesh,str):
        from klampt.io import load
        mesh = load(type='auto', fn=mesh).getTriangleMesh()
    assert vsf_type in ['neural','point']

    aabb = [np.min(mesh.getVertices(), axis=0)-1e-2, np.max(mesh.getVertices(), axis=0)+1e-2]

    vertices_normalized = ((mesh.getVertices() - aabb[0]) / (aabb[1] - aabb[0]) * 2) - 1
    try:
        import mesh2sdf
        print("Computing SDF")
        level = 2 / grid_size
        sdf, _ = mesh2sdf.compute(vertices_normalized, mesh.getIndices(), grid_size, 
                                  fix=True, level=level, return_mesh=True)
        sdf = torch.tensor(sdf, dtype=torch.float32)
        print("Computing SDF Done")
    except ImportError:
        print("Unable to import mesh2sdf, using Klampt SDF conversion instead")
        g = klampt.Geometry3D(mesh)
        g.getTriangleMesh().vertices = vertices_normalized
        res = 2/grid_size
        sdf = g.convert('ImplicitSurface',res)
        sdf = torch.tensor(sdf.getImplicitSurface().values, dtype=torch.float32)

    if vsf_type == 'neural':
        config = NeuralVSFConfig([aabb[0].tolist(),aabb[1].tolist()])
        return NeuralVSF(config, sdf=sdf)
    elif vsf_type == 'point':
        config = PointVSFConfig(bbox=[aabb[0],aabb[1]], voxel_size=1.0/grid_size)
        rest_points = sdf_to_points(sdf, aabb[0], aabb[1], thresh=sdf_thres)        
        return PointVSF(rest_points, config)

def vsf_from_rgbd(rgb_image : np.ndarray, depth_image : np.ndarray, bmin : np.ndarray, bmax : np.ndarray,
                  camera_intrinsics : np.ndarray, camera_extrinsics : np.ndarray,
                  depth_scale : float = 1000.0, depth_trunc : float = 3.0,
                  type = 'point',
                  N_points : int = 30000, voxel_size : float = 0.01) -> Union[NeuralVSF,PointVSF]:
    """Creates an empty point-based VSF from an RGBD image.

    This function creates a VSF from an RGBD image. The occupied volume is defined by the bounding box
    and the volume behind the object (from the camera point of view). The camera intrinsics and extrinsics
    are used to project the depth image into 3D points. The depth scale and truncation are used to convert
    the depth image to meters. The number of points and voxel size are used to create the VSF.

    Inputs:
    - rgb_image: np.ndarray, the RGB image, shape (H,W,3)
    - depth_image: np.ndarray, the depth image, shape (H,W)
    - bmin: np.ndarray, the minimum corner of the bounding box, [x_min, y_min, z_min]
    - bmax: np.ndarray, the maximum corner of the bounding box, [x_max, y_max, z_max]
    - camera_intrinsics: np.ndarray, the camera intrinsics, shape (3,3)
    - camera_extrinsics: np.ndarray, the camera extrinsics, shape (4,4)
    - depth_scale: float, the depth scale, default 1000.0
    - depth_trunc: float, the depth truncation, default 3.0
    - type: str, the type of VSF to create, 'point' or 'neural', default 'point'
    - N_points: int, the maximum number of points to use in the point VSF model, default 30000
    - voxel_size: float, the voxel size, default 0.01
    """
    from klampt import Geometry3D, Heightmap

    vp = klampt.Viewport()
    vp.setPose(*se3.from_homogeneous(camera_extrinsics))
    vp.w, vp.h = rgb_image.shape[1], rgb_image.shape[0]
    vp.fx, vp.fy = camera_intrinsics[0,0], camera_intrinsics[1,1]
    vp.cx, vp.cy = camera_intrinsics[0,2], camera_intrinsics[1,2]

    # remove points beyond the truncation distance    
    depth_image[depth_image > depth_trunc*depth_scale] = 0

    # create a heightmap from the rgbd image
    hm_data = Heightmap()
    hm_data.setViewport(vp)
    hm_data.setHeightImage(depth_image, 1/depth_scale)
    hm_data.setColorImage(rgb_image)
    hm = Geometry3D(hm_data)

    # heightmap to sdf with klampt
    sdf = hm.convert('ImplicitSurface')

    # create a bounding box
    bbox = klampt.GeometricPrimitive()
    bbox.setAABB(bmin, bmax)
    bbox = Geometry3D(bbox)
    bbox = bbox.convert('ImplicitSurface', voxel_size)

    # compute the intersection of the sdf and the bounding box
    sdf.getImplicitSurface().setValues(-sdf.getImplicitSurface().getValues())
    bbox.getImplicitSurface().setValues(-bbox.getImplicitSurface().getValues())
    # use the bbox as the bounding box for the object
    # so merge bbox with sdf
    bbox.merge(sdf)
    sdf = bbox
    sdf.getImplicitSurface().setValues(-sdf.getImplicitSurface().getValues())

    if type == 'point':
        pc = sdf.convert('PointCloud')
        points = pc.getPointCloud().getPoints()

        # farthest point sampling
        # downsample the point cloud in a non-axis aligned way to avoid aliasing
        import open3d as o3d
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        pc_o3d = pc_o3d.farthest_point_down_sample(N_points)
        points = np.asarray(pc_o3d.points)

        return PointVSF(points)
    elif type == 'neural':
        aabb = sdf.getBBTight()
        config = NeuralVSFConfig(aabb=aabb)
        return NeuralVSF(config, sdf=torch.tensor(sdf.getImplicitSurface().getValues(), dtype=torch.float32))

def vsf_from_vsf(config : Union[NeuralVSFConfig,PointVSFConfig], 
                 vsf: Union[NeuralVSF,PointVSF], 
                 convert_config: NeuralVSFEstimatorConfig = NeuralVSFEstimatorConfig()) -> Union[NeuralVSF,PointVSF]:
    """Creates a VSF from a VSF.

    This function can convert between point-based and neural VSFs. The config is used to determine the type of VSF to create.
    Settings in the config and attributes from the source VSF are used to define the new VSF.
    
    Inputs:
    - config: Union[NeuralVSFConfig,PointVSFConfig], the configuration of the VSF
    - vsf: Union[NeuralVSF,PointVSF], the VSF to convert
    - convert_config: NeuralVSFEstimatorConfig, the training configuration for the point to neural conversion
    """

    if isinstance(vsf,PointVSF):
        if isinstance(config,PointVSFConfig):
            # return without any changes
            return vsf
        elif isinstance(config,NeuralVSFConfig):
            # convert to neural VSF
            points = vsf.rest_points
            stiffness = vsf.stiffness

            def compute_pointcloud_voxel_size(points):
                # random pick 1000 points and compute their closest distance to the other points
                num_samples = 1000
                N = points.shape[0]
                idx = torch.randperm(N)[:num_samples]
                dist = torch.norm(points[idx,None,:] - points[None],dim=2)
                # remove self distance
                dist[torch.arange(num_samples), idx] = float('inf')
                dist = torch.min(dist, dim=1).values
                return torch.mean(dist)
            voxel_size = compute_pointcloud_voxel_size(points)
            print("Voxel size: ", voxel_size)

            # build dataloader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(torch.utils.data.TensorDataset(points, stiffness), batch_size=8192, shuffle=True)

            # initialize the neural VSF model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            padding = 0.01
            aabb = torch.stack([points.min(dim=0)[0]-padding, points.max(dim=0)[0]+padding], dim=0).to(device)
            config = NeuralVSFConfig(aabb=aabb)
            vsf2 = NeuralVSF(config)
            vsf2.to(device)

            optimizer = torch.optim.Adam(vsf2.vsfNetwork.get_params(convert_config.lr))
            vsf2.vsfNetwork.train()

            regularizer_samples = convert_config.regularizer_samples
            regularizer_scale = convert_config.regularizer_scale

            from tqdm import tqdm
            for i in tqdm(range(convert_config.max_epochs)):
                for batch in dataloader:
                    points, stiffness = batch
                    stiffness = stiffness.to(device) / voxel_size**3
                    vsf_samples = points.to(device)
                    stiffness2 = vsf2.getStiffness(vsf_samples)
                    loss = torch.mean((stiffness - stiffness2)**2)

                    # regularization term, enforce low stiffness in empty region
                    vsf_samples = torch.rand(regularizer_samples, 3).to(vsf2.device) * (aabb[1] - aabb[0]) + aabb[0]
                    stiffness2 = vsf2.getStiffness(vsf_samples)
                    loss += torch.abs(stiffness2).mean() * regularizer_scale

                    # optimize VSF model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            return vsf2
        else:
            raise ValueError("Invalid config type")
    elif isinstance(vsf,NeuralVSF):
        if isinstance(config,NeuralVSFConfig):
            # return without any changes
            return vsf
        elif isinstance(config,PointVSFConfig):
            # convert to point VSF
            voxel_size = config.voxel_size
            # initialize the point VSF grid, pytorch tensor
            aabb = vsf.config.aabb
            points = torch.meshgrid(*[torch.linspace(aabb[0][i],aabb[1][i],int((aabb[1][i]-aabb[0][i])/voxel_size)) for i in range(3)],indexing='ij')
            points = [dim.reshape(-1) for dim in points]
            points = torch.stack(points,dim=1)

            stiffness = []
            batch_size = 100000
            for i in range(0,points.shape[0],batch_size):
                batch = points[i:i+batch_size].to(vsf.device)
                stiffness.append(vsf.getStiffness(batch).detach().cpu())
            stiffness = torch.cat(stiffness,dim=0)

            # remove low stiffness points
            mask = (stiffness > 1e1).squeeze() # the uniform stiffness is N/m^4, so a large threshold
            points = points[mask]
            stiffness = stiffness[mask]

            vsf2 = PointVSF(rest_points=points, bbox=aabb, voxel_size=voxel_size)
            vsf2.stiffness = stiffness * voxel_size**3
            return vsf2
        else:
            raise ValueError("Invalid config type")
    else:
        raise ValueError("Invalid VSF type")
        