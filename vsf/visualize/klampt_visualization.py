import numpy as np
import torch
import time
from klampt import ImplicitSurface, PointCloud, Geometry3D, GeometricPrimitive
from .. import PointVSF, NeuralVSF
from ..sim import QuasistaticVSFSimulator
from .common import stiffness_to_color, colorize, autodetect_stiffness_values
from typing import Union, Optional, List, Tuple

def implicit_surface_to_levelsets(vg : ImplicitSurface, stiffness_levels: list, cmap : str='YlOrRd') -> Tuple[List[Geometry3D], List[List[float]]]:
    """Gives geometries and colors for each levelset in the implicit surface.
    
    Args:
        vg (ImplicitSurface): the implicit surface
        stiffness_levels (list): the stiffness levels to visualize
        cmap (str, optional): the colormap to use. Defaults to 'YlOrRd'.
    
    Returns:
        Tuple[List[Geometry3D], List[List[float]]]: the list of geometries and colors for each levelset
    """
    from klampt import Geometry3D
    if not isinstance(vg,Geometry3D):
        vg = Geometry3D(vg)
    
    meshes = []
    colors = []
    stiffness_upper = stiffness_levels[-1]
    for i,thresh in enumerate(stiffness_levels):
        m = vg.convert('TriangleMesh',thresh)
        tm = m.getTriangleMesh()
        inds = tm.getIndices()
        #print("Layer",thresh,"has mesh with",len(inds),"triangles")
        if len(inds) > 0:
            #flipped from normal SDF case since higher is inner
            inds[:,1],inds[:,2] = inds[:,2].copy(),inds[:,1].copy()
            #tm.setIndices(inds)
            #sort tris from top to bottom so we don't see back faces
            verts = tm.getVertices()
            order = []
            for j in range(len(inds)):
                c = (verts[inds[j][0]] + verts[inds[j][1]] + verts[inds[j][2]])/3.0
                order.append((-c[2],j))
            order.sort()
            order = [j for (z,j) in order]
            newinds = np.array([inds[j] for j in order])
            tm.setIndices(newinds)

        m = Geometry3D(tm)
        meshes.append(m)
        
        c = stiffness_to_color([thresh],stiffness_upper,cmap)[0]
        opacity = thresh/stiffness_upper if stiffness_upper > 0 else 1
        # TODO: support alternative ways to set opacity
        # opacity = (i+1)/len(stiffness_levels)
        colors.append([c[0],c[1],c[2],opacity])
        # vis.setColor("mesh_"+str(i),c[0],c[1],c[2],t/stiffness_upper)

    return meshes, colors


def visualize_stiffness_volume(vg : ImplicitSurface, stiffness_values : List[float], cmap : str='YlOrRd'):
    """Shows the implicit surface as level sets with the given stiffness values."""
    from klampt import Geometry3D,Appearance
    from klampt import vis
    for i,(m,c) in enumerate(zip(*implicit_surface_to_levelsets(vg,stiffness_values,cmap))):
        a = Appearance()
        a.setColor(*c)
        a.setSilhouette(0)
        a.setCreaseAngle(0)
        vis.add("mesh_"+str(i),m,appearance=a,hide_label=True,draw_order=-i)

    vis.autoFitCamera()
    vis.show()
    while vis.shown():
        time.sleep(0.1)
    vis.scene().clear()
    

def vsfnet_to_volume_grid(vsf_net : torch.nn.Module, aabb : np.ndarray, resolution : float) -> ImplicitSurface:
    vg = ImplicitSurface()
    vg.bmin = aabb[0]
    vg.bmax = aabb[1]
    res = np.ceil((aabb[1]-aabb[0])/resolution).astype(int).tolist()
    values = np.zeros(res)
    for i in range(res[0]):
        #print("Processing slice",i)
        arr = []
        for j in range(res[1]):
            for k in range(res[2]):
                arr.append([i,j,k])
        p = np.array(arr)*resolution + aabb[0]
        s = vsf_net(torch.tensor(p,dtype=torch.float32,device=next(vsf_net.parameters()).device)).detach().cpu().numpy().squeeze()
        values[i,:,:] = s.reshape(res[1],res[2])
    vg.values = values
    return vg

def particle_vsf_to_volume_grid(vsf_samples : np.ndarray, vsf_stiffness : np.ndarray, resolution='auto'):
    """
    Convert a set of samples and stiffness values to a volume grid.
    
    Args:
        vsf_samples (np.ndarray): The sampled points in the VSF, shape (N, 3).
        vsf_stiffness (np.ndarray): The stiffness values of the VSF, shape (N,).
        resolution (float, optional): The resolution of the volume grid. If 'auto', will automatically determine the resolution.
    """
    bbox = [np.min(vsf_samples,axis=0),np.max(vsf_samples,axis=0)]
    if resolution == 'auto':
        mindist = float('inf')
        for i,v in enumerate(vsf_samples):
            if i+1 < len(vsf_samples):
                mindist = min(mindist,np.linalg.norm(v-vsf_samples[i+1]))
        mindist *= np.sqrt(3)
        print("Setting voxel grid resolution",mindist)
        resolution = mindist
    bbox[0] -= resolution*0.5
    bbox[1] += resolution*0.5
    vg = ImplicitSurface()
    vg.bmin = bbox[0]
    vg.bmax = bbox[1]
    dims = np.ceil((bbox[1]-bbox[0])/resolution).astype(int).tolist()
    values = np.zeros(dims)
    counts = np.zeros(dims)
    assert len(vsf_stiffness) == len(vsf_samples)
    for (p,s) in zip(vsf_samples,vsf_stiffness):
        i = tuple(((p-bbox[0])/resolution).astype(int))
        counts[i] += 1
        values[i] += 1.0/counts[i]*(s-values[i])
    vg.values = values 
    return vg


def vsf_to_point_cloud(vsf : Union[PointVSF,NeuralVSF],
                       mask : Optional[Union[str,np.ndarray]]='auto',
                       N_samples = 200000,
                       masked_view_fraction = 0.035,
                       auto_stiffness_threshold = 0.01,
                       feature : str = None, feature_idx = 0) -> PointCloud:
    """See o3d_visualization.vsf_to_point_cloud."""
    from . import o3d_visualization
    from klampt.io import open3d_convert
    return open3d_convert.from_open3d(o3d_visualization.vsf_to_point_cloud(vsf,mask,N_samples,masked_view_fraction,auto_stiffness_threshold,feature,feature_idx))


def vsf_to_level_sets(vsf : Union[PointVSF,NeuralVSF],
                      stiffness_values : Union[List[float],str]= 'auto',
                      feature : str = None, feature_idx : int = 0,
                      cmap : str = 'YlOrRd') -> Tuple[List[Geometry3D], List[List[float]]]:
    """Returns a list of geometries and colors for each levelset in the VSF
    volume visualization."""
    if isinstance(vsf, NeuralVSF):
        neural_res = 0.005
        return implicit_surface_to_levelsets(vsfnet_to_volume_grid(vsf.vsfNetwork,vsf.getBBox().cpu().numpy(),neural_res),stiffness_values,cmap)
    else:
        assert isinstance(vsf,PointVSF)
        assert isinstance(feature_idx,int)
        if feature is not None:
            values = vsf.features[feature]
            if values.ndim == 2:
                values = values[:,feature_idx]
        else:
            values = vsf.stiffness
        return implicit_surface_to_levelsets(particle_vsf_to_volume_grid(vsf.rest_points.cpu().numpy(),values.cpu().numpy()),stiffness_values,cmap)


def vsf_show(vsf : Union[PointVSF,NeuralVSF],
             stiffness_values : Union[List[float],str]= 'auto',
             type : str = 'auto',
             feature : str = None, feature_idx : Union[int,Tuple[int]] = 0,
             cmap : str = 'YlOrRd'):
    """Visualizes a VSF using a Klamp't visualization window.

    By default, shows the stiffness field.  If feature is not None, shows the
    feature field instead.

    Args:
        vsf (Union[PointVSF,NeuralVSF]): the VSF to visualize
        stiffness_values (Union[List[float],str], optional): the stiffness values to visualize.  If 'auto', will
            automatically determine the stiffness values. Defaults to 'auto'.
        type (str, optional): the type of visualization.  Can be 'auto', 'points', or 'volume'. Defaults to 'auto'.
        feature (str, optional): the feature to visualize. Defaults to None.
        feature_idx (Union[int,Tuple[int]], optional): the feature index to visualize.
            Defaults to 0. If a tuple is given, these channels will be used directly as colors.
        cmap (str, optional): the colormap to use. Defaults to 'YlOrRd'.  Can also be 'random' for random
            colors assigned to each distinct value of the feature (e.g. a segment index).

    """
    from klampt import vis

    vis.setBackgroundColor(1,1,1)

    aabb = vsf.getBBox().cpu().numpy().astype(float)
    if isinstance(vsf, NeuralVSF):
        neural_res = 0.005
        nvg = vsfnet_to_volume_grid(vsf.vsfNetwork,aabb,neural_res)

        if isinstance(stiffness_values,str) and stiffness_values == 'auto':
            #correction of 20 ^ (1/3) is used to scale between 1k and 20k
            stiffness_neural = nvg.getValues().flatten()
            lb = 1e2
            stiffness_values = autodetect_stiffness_values((stiffness_neural),lb=lb,scales=[1.0]).tolist()
            print("vsf_show: Using the following stiffness values:",stiffness_values)
        
        if type == 'points':
            pc = vsf_to_point_cloud(vsf,mask='auto')
            geom = Geometry3D(pc)
            vis.add("pc",geom,hide_label=True,pointSize=5.0)
            vis.autoFitCamera()
            vis.show()
            while vis.shown():
                time.sleep(0.1)
            vis.scene().clear()
        else:
            visualize_stiffness_volume(nvg,stiffness_values, cmap=cmap)
    else:
        #consider alternative features
        point_colors = None
        if feature is not None:
            values = vsf.features[feature]
            if values.ndim == 2:
                if isinstance(feature_idx,(tuple,list)): 
                    assert type == 'auto' or type == 'points',"Can't colorize volumes"
                    type = 'points'
                    point_colors = [values[:,i].cpu().numpy() for i in feature_idx]
                    while len(point_colors) < 3:
                        point_colors.append(np.zeros(len(values)))
                    #normalize and resize to N x 3
                    for i in range(len(point_colors)):
                        c = point_colors[i]
                        if c.min() == c.max():
                            c = np.zeros(len(c))
                        else:
                            c = (c - c.min())/(c.max()-c.min())
                        point_colors[i] = c
                    point_colors = np.array(point_colors).T
                else:
                    values = values[:,feature_idx]
        else:
            values = vsf.stiffness
        values = values.cpu().numpy()
        if values.min() == values.max():
            if type == 'auto':
                type = 'points'
            elif type == 'volume':
                print("vsf_show: Warning: VSF has constant stiffness value",values[0])
        if type == 'points':
            #show as particle grid
            pc = PointCloud()
            pc.points = vsf.rest_points.cpu().numpy()
            if point_colors is None:
                point_colors = colorize(values,cmap=cmap)
            pc.setColors(point_colors,('r','g','b'))
            geom = Geometry3D(pc)
            vis.add("pc",geom,hide_label=True,pointSize=5.0)
            vis.autoFitCamera()
            vis.show()
            while vis.shown():
                time.sleep(0.1)
            vis.scene().clear()
            return
        else:
            nvg = particle_vsf_to_volume_grid(vsf.rest_points.cpu().numpy(),values)
            if isinstance(stiffness_values,str) and stiffness_values == 'auto':
                stiffness = values
                lb = 0.0
                stiffness_values = autodetect_stiffness_values((stiffness),lb=lb,scales=[1.0]).tolist()
                print("vsf_show: Using the following stiffness values:",stiffness_values)
        
            visualize_stiffness_volume(nvg,stiffness_values, cmap =cmap)



def create_line_set(start_points:np.ndarray, end_points:np.ndarray) -> list[Geometry3D]:
    line_lst = []
    start_points = start_points.copy()
    end_points = end_points.copy()
    for start_pt, end_pt in zip(start_points, end_points):
        line = Geometry3D()
        geom_prim = GeometricPrimitive()
        geom_prim.setSegment(start_pt, end_pt)
        line.setGeometricPrimitive(geom_prim)
        line_lst.append(line)
    return line_lst


def add_sim_to_vis(sim : QuasistaticVSFSimulator,
                   draw_normals = True,
                   draw_springs=True) -> List[str]:
    """Adds a QuasistaticVSFSimulator to the Klampt visualization window.
    
    Suggest using vis.lock() / vis.unlock() around this call.

    Returns all the names of the items added to the visualization.
    """
    from klampt import vis
    from klampt.io import numpy_convert
    from klampt.math import se3,so3,vectorops
    from klampt.model.geometry import TriangleMesh
    from ..sim.point_vsf_body import PointVSFQuasistaticSimBody
    from ..sim.neural_vsf_body import NeuralVSFQuasistaticSimBody

    visitems = []
    visitems.append("world")
    vis.add("world",sim.klampt_world_wrapper.world)
    for name,obj in sim.vsf_objects.items():
        if isinstance(obj,PointVSFQuasistaticSimBody):
            point_cloud = obj.deformed_points()
            point_cloud_data = numpy_convert.from_numpy(point_cloud,'PointCloud')  # type: PointCloud
            colors = stiffness_to_color(obj.vsf_model.stiffness.cpu().numpy())
            colors = np.append(colors,np.full((len(colors),1),0.25),axis=1)
            colors[obj.point_object_idx >= 0,3] = 1.0
            colors[obj.point_object_idx >= 0,0],colors[obj.point_object_idx >= 0,2] = colors[obj.point_object_idx >= 0,2],colors[obj.point_object_idx >= 0,0]
            point_cloud_data.setColors(colors,color_format=('r','g','b','a'))
            pcg = Geometry3D(point_cloud_data)
            visitems.append(name)
            vis.add(name,pcg)

            for i in range(len(obj.point_object_idx)):
                if obj.point_object_idx[i] >= 0:
                    body = obj.point_object_idx[i]
                    T = sim.klampt_world_wrapper.bodies_dict[sim.klampt_world_wrapper.name_lst[body]].getTransform()
                    pt = se3.apply(T,obj.anchor_local[i])
                    if draw_normals or draw_springs:
                        visitems.append(name+'_cp'+str(i))
                    #draw connections to rest springs
                    if draw_springs:
                        rest_pt = se3.apply(se3.from_ndarray(obj.pose),obj.vsf_model.rest_points[i].cpu().numpy())
                        vis.add(visitems[-1],[pt,rest_pt],color=(0,0,1,1),hide_label=True)
                    #draw normals
                    if draw_normals:
                        nm = so3.apply(T[0],obj.anchor_normal_local[i])
                        vis.add(visitems[-1],[pt,vectorops.madd(pt,nm,0.02)],color=(0,0,1,1),hide_label=True)
        else:
            assert isinstance(obj,NeuralVSFQuasistaticSimBody)
            print("TODO: draw neural VSF info")
    return visitems