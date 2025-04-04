from .base_sensor import BaseSensor
from .base_calibrator import BaseCalibrator,TareCalibrator
from dataclasses import dataclass
from typing import List,Optional

@dataclass
class SensorConfig:
    name : str
    type : str
    attachment_name : str                     # the rigid object, link, or robot this should be attached to
    link_names : Optional[List[str]] = None   # if it's a robot, the links of the robot that are affected

def sensor_from_config(config : SensorConfig):
    if config.type == 'PunyoDenseForceSensor':
        from .punyo_dense_force_sensor import PunyoDenseForceSensor
        return PunyoDenseForceSensor(config.name, config.attachment_name)
    if config.type == 'PunyoPressureSensor':
        from .punyo_pressure_sensor import PunyoPressureSensor
        return PunyoPressureSensor(config.name, config.attachment_name)
    if config.type == 'JointTorqueSensor':
        from .joint_torque_sensor import JointTorqueSensor
        assert config.link_names is not None
        return JointTorqueSensor(config.name, config.attachment_name, config.link_names)
    else:
        raise NotImplementedError("Sensor type {} not implemented".format(config.type))
    
@dataclass
class CalibrationConfig:
    type : str
    num_samples : int = '10'
    output_key : str = 'tare'

def calibration_from_config(config : CalibrationConfig):
    if config.type == 'tare':
        return TareCalibrator(config.num_samples, config.output_key)
    else:
        raise NotImplementedError("Calibration type {} not implemented".format(config.type))
