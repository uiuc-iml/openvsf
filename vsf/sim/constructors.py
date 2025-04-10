import numpy as np
from klampt.math import se3
import open3d as o3d
from .quasistatic_sim import QuasistaticVSFSimulator, ContactParams
from .neural_vsf_body import NeuralVSFSimConfig
from .klampt_world_wrapper import klamptWorldWrapper
from ..sensor.constructors import SensorConfig, sensor_from_config
from ..core.point_vsf import PointVSF,PointVSFConfig
from ..core.neural_vsf import NeuralVSF,NeuralVSFConfig
from ..core.vsf_factory import VSFRGBDCameraFactoryConfig
from ..constructors import AnyVSFConfig, vsf_from_config, vsf_from_file
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union

@dataclass
class WorldObjectConfig:
    """Configures a single object in the world.  `type` can be
    "robot", "rigid", or "deformable".

    The object may be attached to another object, and may have an
    initial configurations.
    """
    type : str
    file_name : str
    parent_name : Optional[str] = None
    parent_relative_transform : Optional[List[List]] = None
    initial_config : Optional[np.ndarray] = None


@dataclass
class WorldConfig:
    """Configures a world with a set of objects.
    """
    objects : Dict[str,WorldObjectConfig] = field(default_factory=dict)


def world_wrapper_from_config(config : WorldConfig) -> klamptWorldWrapper:
    """
    Create a Klamp't world wrapper from a configuration.
    """
    world_wrapper = klamptWorldWrapper()
    for name,item in config.objects.items():
        item_type = item.type
        if item_type == 'robot':
            world_wrapper.add_robot(name, item.file_name)
            if item.initial_config is not None:
                world_wrapper.world.robot(name).setConfig(item.initial_config)
        elif item_type == 'rigid':
            parent_name = item.parent_name
            parent_relative_transform = np.array(item.parent_relative_transform)
            world_wrapper.add_geometry_from_file(name, item.file_name, 'rigid', parent_name=parent_name, parent_relative_transform=parent_relative_transform)
            if item.initial_config is not None:
                world_wrapper.bodies_dict[name].setTransform(se3.ndarray(item.initial_config))
        elif item_type == 'deformable':
            parent_name = item.parent_name
            parent_relative_transform = np.array(item.parent_relative_transform)
            world_wrapper.add_geometry_from_file(name, item.file_name, 'deformable', parent_name=parent_name, parent_relative_transform=parent_relative_transform)
            if item.initial_config is not None:
                world_wrapper.bodies_dict[name].setTransform(se3.ndarray(item.initial_config))
    return world_wrapper


@dataclass
class VSFBodyConfig:
    """Configures a VSF body in a simulator."""
    model : AnyVSFConfig
    config : Optional[Union[NeuralVSFSimConfig,ContactParams]] = None
    pose : Optional[List[list]] = None


@dataclass
class SimulatorConfig:
    """Configures a simulator"""
    type : str = 'quasistatic'
    world_pcd_samples : int = 1000
    world_sdf_resolution : float = 0.01
    deformables : Dict[str,VSFBodyConfig] = field(default_factory=dict)


def simulator_from_config(config : SimulatorConfig, world_config:WorldConfig, sensors : Dict[str,SensorConfig] = {}) -> QuasistaticVSFSimulator:
    assert config.type == 'quasistatic'
    klampt_wrapper = world_wrapper_from_config(world_config)
    sensor_list = []
    for sensor_name, sensor_config in sensors.items():
        sensor = sensor_from_config(sensor_config)
        sensor_list.append(sensor)

    # Simulation contact parameters 
    vsf_sim = QuasistaticVSFSimulator(klampt_wrapper, sensors=sensor_list)
    for name,item in config.deformables.items():
        vsf_model = vsf_from_config(item.model)
        vsf_obj = vsf_sim.add_deformable(name,vsf_model,item.config)
        if item.pose is not None:
            vsf_obj.pose = np.array(item.pose)

    klampt_wrapper.setup_local_pcd_lst(sample_backend='open3d', num_of_pts=config.world_pcd_samples)
    klampt_wrapper.setup_local_sdf_lst(resolution=config.world_sdf_resolution)
    return vsf_sim