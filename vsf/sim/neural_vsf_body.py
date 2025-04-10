from .. import NeuralVSF
from .klampt_world_wrapper import klamptWorldWrapper
import torch
import klampt
from klampt.math import se3
import numpy as np
from typing import Tuple,Union
from dataclasses import dataclass
from ..utils.perf import DummyRecorder

def compute_vertex_normal(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute vertex normal from triangle mesh.
    
    Args: 
        points: Tensor of shape (N, 3) representing the vertices of the mesh.
        triangles: Tensor of shape (M, 3) representing the indices of the vertices that form each triangle.
    Returns:
        vertex_normal: Tensor of shape (N, 3) representing the normal vector at each vertex.
    """
    points_tri = points[triangles]
    vertex_normal = torch.zeros_like(points)
    a = points_tri[:, 1] - points_tri[:, 0]
    b = points_tri[:, 2] - points_tri[:, 0]
    area_v = torch.linalg.cross(a, b) / 2
    vertex_normal.index_add_(0, triangles.reshape(-1), area_v[:, None, :].repeat(1, 3, 1).reshape(-1, 3) / 3)
    vertex_normal /= (torch.linalg.norm(vertex_normal, dim=1, keepdim=True) + 1e-10)
    return vertex_normal

def compute_vertex_area(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute vertex area from triangle mesh.
    
    Args:
        points: Tensor of shape (N, 3) representing the vertices of the mesh.
        triangles: Tensor of shape (M, 3) representing the indices of the vertices that form each triangle.
    Returns:
        vertex_area: Tensor of shape (N,) representing the area at each vertex.
    """
    points_tri = points[triangles]
    vertex_area = torch.zeros(len(points), dtype=points.dtype, device=points.device)
    a = points_tri[:, 1] - points_tri[:, 0]
    b = points_tri[:, 2] - points_tri[:, 0]
    area_v = torch.linalg.cross(a, b) / 2
    area_v = torch.linalg.norm(area_v, dim=1)
    vertex_area.index_add_(0, triangles.reshape(-1), area_v[:, None].repeat(1, 3).reshape(-1) / 3)
    return vertex_area

def compute_vsf_force(vertices: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
                      vertices_normal: torch.Tensor, vsf: NeuralVSF,
                      N_samples=100):
    """
    Compute the VSF force on the vertices of the mesh.

    Args:
        vertices: Tensor of shape (N, 3) representing the current vertices position of the mesh.
        start: Tensor of shape (N, 3) representing the start points of the line segments.
        end: Tensor of shape (N, 3) representing the end points of the line segments.
        vertices_normal: Tensor of shape (N, 3) representing the normal vectors at each vertex.
        vsf: NeuralVSF object representing the volumetric stiffness field.
        N_samples: Number of samples to use for VSF integration for each segment.
    Returns:
        vsf_force_all: Tensor of shape (N, 3) representing the VSF force on each vertex.
    """
    sample_density = 1e3

    vsf_force_all = torch.zeros_like(vertices)

    mask = (vertices_normal * (vertices - start)).sum(dim=1) > 0
    vertices = vertices[mask]
    start = start[mask]
    end = end[mask]
    vertices_normal = vertices_normal[mask]

    # generate samples between vertices and contact positions
    dist = torch.linalg.norm(end - start, dim=1)
    near, far = torch.zeros_like(dist), dist
    t_vals = torch.linspace(0., 1., steps=N_samples, device=vertices.device)
    z_vals = near[:,None] * (1.-t_vals[None,:]) + far[:,None] * (t_vals[None,:])

    mids = .5 * (z_vals[:,1:] + z_vals[:,:-1])
    upper = torch.cat([mids, z_vals[:,-1:]], -1)
    lower = torch.cat([z_vals[:,:1], mids], -1)
    # samples points in those intervals
    t_rand = torch.rand(z_vals.shape, device=vertices.device)
    z_vals = lower + (upper - lower) * t_rand
    direction = (end - start) / \
                (torch.linalg.norm(end - start, dim=1, keepdim=True) + 1e-6)
    samples = start[:,None,:] + z_vals[...,None] * direction[:,None,:]
    samples = samples + torch.randn_like(samples) * 0.0

    # compute forces
    reference_N_samples = dist * sample_density
    stiffness = vsf(samples)
    vsf_force = torch.sum((samples - vertices[:,None,:]).double() * stiffness.double(), dim=1) * \
                reference_N_samples[:, None] / N_samples
    
    vsf_force *= (vertices_normal.double() * direction.double()).sum(dim=1, keepdim=True)
    vsf_force = vsf_force.float()

    vsf_force_all[mask] = vsf_force

    return vsf_force_all


@dataclass
class NeuralVSFSimConfig:
    """Describes how the NeuralVSF simulator should behave.
    
    Attributes:
        N_samples: Number of samples to use for VSF integration.
    """
    N_samples: int = 100
    

class NeuralVSFQuasistaticSimBody:
    """A class that simulates the quasistatic behavior of a body
    using NeuralVSF.
    
    As object points are dragged through the volume, we record
    the forces on each point and integrate them to get the
    final force on the object.

    TODO: implement this in recursive form.
    """
    def __init__(self, vsf: NeuralVSF, config: NeuralVSFSimConfig):
        assert isinstance(vsf, NeuralVSF), 'NeuralVSFSimulator only supports NeuralVSF as the VSF model.'
        self.vsf = vsf
        box = klampt.GeometricPrimitive()
        box.setAABB(list(vsf.getBBox()[0]), list(vsf.getBBox()[1]))
        self.bbox = klampt.Geometry3D(box)
        self.pose = np.eye(4)

        self.config = config
        # NOTE: for consistency with point vsf body, add a dummy recorder
        self.perfer = DummyRecorder()

        self.reset()

    def reset(self):
        """
        Setup contact information recorder and force integrators.
        
        This function can be called to clear force integration history of NeuralVSF.
        Update 
        `self.vertex_trajectory_local`: dict[str, torch.Tensor], keys are object name and 
        save all triangle mesh vertices trajectory in contact with NeuralVSF in the VSF local frame.
        `self.vertex_normal_local`: dict[str, torch.Tensor], keys are object name and 
        save all vertices normal corresponding to the vertices trajectory in the VSF local frame.
        """
        self.vertex_trajectory_local = {}   #previous locations of object vertices, in local frame
        self.vertex_normal_local = {}   #previous normal of object vertices, in local frame
        self.vertex_contact_mask = {} #previous mask for contact vertices
        
        self.contact_mesh = {} # cache for contact object mesh

    def step(self, state : klamptWorldWrapper, dt : float) -> Tuple[np.ndarray,np.ndarray, torch.Tensor, torch.Tensor]:
        """
        This function runs NeuralVSF simulation based on the control sequence.
        
        Returns:
            obj_index, obj_elems, cps, forces: contains the object index,
            element index, contact point, and contact force (world coordinates)
            for each contact.
        """         
        from klampt.model.collide import bb_intersect
        self.bbox.setCurrentTransform(*se3.from_ndarray(self.pose))
        bbw = self.bbox.getBBTight()
        obj_index = []
        obj_elems = []
        cps = []
        forces = []
        for i,objectName in enumerate(state.name_lst):
            body = state.bodies_dict[objectName]
            obbw = body.geometry().getBB()
            if not bb_intersect(bbw, obbw):
                # no collision, remove from tracking
                self.vertex_trajectory_local[objectName] = []
                self.vertex_normal_local[objectName] = []
                self.vertex_contact_mask[objectName] = []
                continue
            
            # check if the object is in cache
            # if the mesh is not in cache or the object is deformable, load the mesh to device
            if objectName not in self.contact_mesh or state.control_type_dict.get(objectName, '') == 'deformable':
                tmesh = body.geometry().getTriangleMesh()
                vertices = torch.tensor(tmesh.getVertices(), dtype=self.dtype, device=self.device)
                triangles = torch.tensor(tmesh.getIndices(), device=self.device)
                self.contact_mesh[objectName] = (vertices, triangles)
            # if the object is in cache, use the cached mesh
            # can potentially reduce the data copying
            else:
                vertices, triangles = self.contact_mesh[objectName]

            o2w = torch.tensor(se3.homogeneous(body.getTransform()), dtype=self.dtype, device=self.device)
            l2w = torch.tensor(self.pose, dtype=self.dtype, device=self.device)
            w2l = torch.linalg.inv(l2w)

            vertices = vertices @ o2w[:3, :3].T + o2w[:3, 3] # object frame to world frame
            vertices = vertices @ w2l[:3, :3].T + w2l[:3, 3] # world frame to vsf local frame

            # compute mesh normal and area
            vertices_normal = compute_vertex_normal(vertices, triangles)
            vertices_normal *= compute_vertex_area(vertices, triangles)[:, None]
            
            # compute contact mask
            vertices_mask = torch.ones(len(vertices), dtype=torch.bool, device=self.device)
            if self.vsf.vsfNetwork.sdf is not None:
                vertices_mask = self.vsf.vsfNetwork.get_sdf(vertices) < 0

            if objectName not in self.vertex_trajectory_local:
                self.vertex_trajectory_local[objectName] = []
                self.vertex_normal_local[objectName] = []
                self.vertex_contact_mask[objectName] = []
            self.vertex_trajectory_local[objectName].append(vertices)
            self.vertex_normal_local[objectName].append(vertices_normal)
            self.vertex_contact_mask[objectName].append(vertices_mask)

            N = len(self.vertex_trajectory_local[objectName]) - 1 # number of line segments
            if N == 0:
                continue
            start = []
            end = []
            # check if the vertices are in contact with the vsf
            any_contact = torch.zeros(vertices.shape[0], dtype=torch.bool, device=self.device)
            for j in range(len(self.vertex_trajectory_local[objectName])-1):
                s = self.vertex_trajectory_local[objectName][j]
                e = self.vertex_trajectory_local[objectName][j+1]

                start.append(s)
                end.append(e)
                any_contact = any_contact | self.vertex_contact_mask[objectName][j]
                
            if any_contact.sum() == 0:
                continue
                
            start = torch.stack(start) # NM x 3, M: number of vertices
            end = torch.stack(end) # NM x 3, M: number of vertices
            
            # no need to compute force for vertices that are not in contact with the vsf at any time step
            start = start[:, any_contact].reshape(-1, 3) # NM' x 3, M': number of vertices in contact
            end = end[:, any_contact].reshape(-1, 3) # NM' x 3, M': number of vertices in contact
            vertices_contact = vertices[any_contact]
            vertices_normal_contact = vertices_normal[any_contact]
            
            vertices_repeat = vertices_contact.repeat(N, 1).view(-1, 3) # NM' x 3, M': number of vertices in contact
            vertices_normal_repeat = vertices_normal_contact.repeat(N, 1).view(-1, 3) # NM' x 3, M': number of vertices in contact
            
            vertices_force = torch.zeros_like(vertices)
            force = -compute_vsf_force(vertices_repeat, start, end, vertices_normal_repeat,
                                       self.vsf.vsfNetwork,
                                       self.config.N_samples // N + 1) # split samples to each segment
            vertices_force[any_contact] = force.reshape(N, -1, 3).sum(dim=0) # in vsf local frame
            vertices_force = vertices_force @ l2w[:3, :3].T # vsf local frame to world frame

            obj_index.append([i]*len(vertices_force))
            obj_elems.append(np.arange(len(vertices_force)))
            cps.append(vertices)
            forces.append(vertices_force)

        if obj_index == []:
            return np.empty(0), np.empty(0), torch.empty(0, 3), torch.empty(0, 3)

        return np.concatenate(obj_elems), np.concatenate(obj_index), torch.concat(cps), torch.concat(forces)

    def state(self):
        return {
            'pose': self.pose,
            'trajectory_vertex': self.vertex_trajectory_local,
            'trajectory_normal': self.vertex_normal_local
        } 
    
    def load_state(self, state: dict):
        self.pose = state['pose']
        self.vertex_trajectory_local = state['trajectory_vertex']
        self.vertex_normal_local = state['trajectory_normal']

    @property
    def device(self):
        return self.vsf.device
    
    @property
    def dtype(self):
        return self.vsf.dtype