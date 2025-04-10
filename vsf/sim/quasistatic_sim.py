from .klampt_world_wrapper import klamptWorldWrapper
from klampt.math import se3
from ..utils.perf import PerfRecorder, DummyRecorder
from ..sensor.base_sensor import BaseSensor, SimState, ContactState
from .. import PointVSF,NeuralVSF
from .point_vsf_body import PointVSFQuasistaticSimBody, ContactParams
from .neural_vsf_body import NeuralVSFQuasistaticSimBody,NeuralVSFSimConfig
import torch
import numpy as np
from typing import Union
from dataclasses import asdict

class QuasistaticVSFSimulator:
    """
    A simulator for Volumetric Stiffness Fields that assumes quasistatic,
    stick-slip contact with PointVSF objects and sticking-only contact
    with NeuralVSF objects.

    Usage:
    
    .. code-block:: python
    
        world = klamptWorldWrapper(klampt_world)
        robot = klampt_world.robot(0)
        sensors = [JointTorqueSensor('torques',robot.name)]
        sim = QuasistaticVSFSimulator(world, sensors)
        sim.add_deformable('vsf_name',vsf)
        dt = 0.1
        for i in range(30):
            config = robot.getConfig()
            config[3] += dt * 1.0
            control = {robot.name:config}  #moves the robot's 3rd joint
            sim.step(control, dt)
            print(sim.measurements())
            #print(sim.state())
            #print(sim.measurement_jacobians())
    
    Attributes:
        klampt_world_wrapper (klamptWorldWrapper): 
            the Klamp't world wrapper

        vsf_objects (dict[str,Union[PointVSF,NeuralVSF]]): 
            the VSF objects

        sensors (list[BaseSensor]): 
            the sensors attached to the simulator

        perfer (PerfRecorder): 
            a performance recorder, default DummyRecorder
        
    """

    def __init__(self, klampt_world_wrapper:klamptWorldWrapper, sensors:list[BaseSensor]):
        self.klampt_world_wrapper = klampt_world_wrapper
        self.vsf_objects : dict[str, Union[PointVSFQuasistaticSimBody,NeuralVSFQuasistaticSimBody]] = {} 
        self.sensors = sensors
        #attach sensors to objects
        for sensor in self.sensors:
            if sensor.attachModelName in self.klampt_world_wrapper.world.getRobotsDict():
                rob = self.klampt_world_wrapper.world.robot(sensor.attachModelName)
                sensor.attach(rob)
            else:
                try:
                    obj = self.klampt_world_wrapper.world.rigidObject(sensor.attachModelName)
                except Exception as e:
                    raise ValueError("Unable to find sensor {} attachment body {}".format(sensor.name, sensor.attachModelName)) from e
                assert obj.id >= 0, "Unable to find sensor {} attachment body {}".format(sensor.name, sensor.attachModelName)
                sensor.attach(obj)
        self._state = SimState(body_transforms={},body_states={},contacts={})
        self.perfer = PerfRecorder()
    
    def set_perfer(self, perfer:PerfRecorder):
        self.perfer = perfer

    def add_deformable(self,
                       name:str,
                       vsf:Union[PointVSF,NeuralVSF],
                       contact_params : ContactParams = None) -> Union[PointVSFQuasistaticSimBody,NeuralVSFQuasistaticSimBody]:
        if contact_params is None:
            contact_params = ContactParams()
        if name in self.vsf_objects:
            raise ValueError("Object {} already exists in the simulator".format(name))
        if name in self.klampt_world_wrapper.bodies_dict:
            raise ValueError("Object {} already exists in the world".format(name))
        if isinstance(vsf, NeuralVSF):
            config = NeuralVSFSimConfig()
            self.vsf_objects[name] = NeuralVSFQuasistaticSimBody(vsf, config)
        else:
            self.vsf_objects[name] = PointVSFQuasistaticSimBody(vsf, contact_params)
        
        self.vsf_objects[name].perfer = self.perfer
        
        return self.vsf_objects[name]

    def get_sensor(self, name:str) -> BaseSensor:
        """
        Returns the sensor with the given name, return None
        if the sensor does not exist.
        
        Inputs:
        - name (str): the name of the sensor
        
        Outputs:
        - sensor (BaseSensor): the sensor object
        """
        for sensor in self.sensors:
            if sensor.name == name:
                return sensor
        return None

    def reset(self):
        """
        This function resets the simulator state to zero state.
        """
        for vsf in self.vsf_objects.values():
            vsf.reset()

    def state(self) -> SimState:
        """
        Return the internal state of the simulator.
        """
        return self._state

    def load_state(self, state : SimState):
        """
        Load the internal state of the simulator.
        """
        self._state = state
        for k,T in state.body_transforms.items():
            if k not in self.vsf_objects:
                self.klampt_world_wrapper.bodies_dict[k].setTransform(*se3.from_ndarray(T.cpu().numpy()))
            else:
                # TODO: unify pose torch/numpy data types
                if isinstance(T, torch.Tensor):
                    T = T.cpu().numpy()
                self.vsf_objects[k].pose = T
        for k,kstate in state.body_states.items():
            if k in self.vsf_objects:
                self.vsf_objects.load_state(kstate)
            else:
                r = self.klampt_world_wrapper.world.robot(k)
                if r.id < 0:
                    raise ValueError("Robot / vsf object {} not found in the world".format(k))
                r.setConfig(kstate.cpu().numpy().tolist())

    def state_dict(self) -> dict:
        """
        This function returns the current state of the simulator.

        The dictionary has strings as keys and can contain lists,
        numpy arrays, and torch.Tensors as values.
        """
        return asdict(self._state)
        
    def load_state_dict(self, state : dict):
        """
        This function loads the simulator state from a dictionary 
        previously returned by state_dict().
        """
        self.load_state(SimState(body_transforms=state['body_transforms'],body_states=state['body_states'],contacts={k:ContactState(**v) for (k,v) in state['contacts'].items()}))
    
    def step(self, controls: dict[str, np.ndarray], dt : float): 
        """
        Simulates the state transition based on the control input.

        Args:
            controls (dict[str, np.ndarray]): A dictionary containing the control inputs. 
                The keys are the names of the control inputs. Each control input is 
                a NumPy array of shape ``(*controlInputShape)``.
                For example, with a single robot arm in the simulator, the control input 
                shape is ``(numJoints,)``.
            
            dt (float): The time step for the simulation.

        """
        if len(self.vsf_objects) == 0:
            # no VSF objects, use default device and dtype
            tsr_params = {'device': torch.device('cpu'), 'dtype': torch.float32}
        else:           
            tsr_params_set = set([(v.device, v.dtype) for v in self.vsf_objects.values()])
            # Error if VSFs have different devices or dtypes
            if len(tsr_params_set) > 1:
                for k, v in self.vsf_objects.items():
                    print(f"Object {k:<10} has device {str(v.device):<10} and dtype {str(v.dtype):<10}")
                raise ValueError("All VSF objects must have the same device and dtype")
            tsr_params = {'device': list(tsr_params_set)[0][0], 'dtype': list(tsr_params_set)[0][1]}
        
        self.perfer.start('kinematic')        
        for name, control in controls.items():
            if name in self.klampt_world_wrapper.control_type_dict:
                self.klampt_world_wrapper.apply_control(name, control)
            elif name in self.vsf_objects:
                assert control.shape == (4,4),"Can only accept VSF pose controls for now"
                self.vsf_objects[name].pose = control
            else:
                print("Warning: control {} not found in the simulator".format(name))
        self.perfer.stop('kinematic')
        
        self.perfer.start('get_robot_configs')
        self._state = SimState(body_transforms={},body_states={},contacts={})
        for i in range(self.klampt_world_wrapper.world.numRobots()):
            robot = self.klampt_world_wrapper.world.robot(i)
            self._state.body_states[robot.name] = torch.tensor(robot.getConfig(), **tsr_params)
        self.perfer.stop('get_robot_configs')
        
        self.perfer.start('get_body_transforms')
        for (k,v) in self.klampt_world_wrapper.get_geom_trans_dict(format='numpy').items():
            self._state.body_transforms[k] = torch.tensor(v, **tsr_params)
        self.perfer.stop('get_body_transforms')

        self.perfer.start('get_vsf_transforms')
        #VSF objects are separate        
        for (k,v) in self.vsf_objects.items():
            self._state.body_transforms[k] = torch.tensor(v.pose, **tsr_params)
        self.perfer.stop('get_vsf_transforms')
        
        #TODO: should we update the particle positions in _state.body_states?

        self.perfer.start('update_vsf_states')
        #do contact detection and reponse for VSF objects in contact
        for k,o in self.vsf_objects.items():
            self.perfer.start('vsf_step')
            res = o.step(self.klampt_world_wrapper, dt)
            self.perfer.stop('vsf_step')
            
            self.perfer.start('update_contact_states')
            if isinstance(o,PointVSFQuasistaticSimBody):
                vsf_pt_idx, body_idx, cps, forces = res
                assert isinstance(forces,torch.Tensor)
                bodies = np.unique(body_idx)
                for b in bodies:
                    bname = self.klampt_world_wrapper.name_lst[b]
                    inds = (body_idx == b).nonzero()[0]
                    csk = ContactState(torch.tensor(cps[inds], **tsr_params), forces[inds], 
                                       torch.tensor(vsf_pt_idx[inds], device=tsr_params['device']), None)
                    csb = ContactState(csk.points, -csk.forces, csk.elems2, csk.elems1)
                    self._state.contacts[(k,bname)] = csk
                    self._state.contacts[(bname,k)] = csb
            else:
                #Neural VSF returns the body point index rather than the VSF point index
                body_pt_idx, body_idx, cps, forces = res
                assert isinstance(forces,torch.Tensor)
                bodies = np.unique(body_idx)
                for b in bodies:
                    bname = self.klampt_world_wrapper.name_lst[b]
                    inds = (body_idx == b).nonzero()[0]
                    assert isinstance(cps,torch.Tensor)
                    csk = ContactState(cps[inds], forces[inds], None, body_pt_idx[inds])
                    csb = ContactState(csk.points, -csk.forces, csk.elems2, csk.elems1)
                    self._state.contacts[(k,bname)] = csk
                    self._state.contacts[(bname,k)] = csb
            self.perfer.stop('update_contact_states')
            
        self.perfer.stop('update_vsf_states')
        
        #do sensor updates
        self.perfer.start('sensor')
        for sensor in self.sensors:
            sensor.update(self._state)
        self.perfer.stop('sensor')

    def measurements(self) -> dict[str, torch.Tensor]:
        """Returns the simulated noise-free measurements at the current state"""
        observation_dict = {}
        for sensor in self.sensors:
            observation_dict[sensor.name] = sensor.predict(self._state)
        return observation_dict

    def get_control_keys(self, suffix='_state') -> dict[str, str]:
        """
        Returns the control keys of the simulator.
        
        Args: 
            suffix: String to be added after the object name, default '_state'.
        """
        control_keys = {}
        for name in self.klampt_world_wrapper.control_type_dict.keys():
            control_keys[name] = name + suffix
        for name in self.vsf_objects.keys():
            control_keys[name] = name + suffix
        return control_keys
    
    def get_sensor_keys(self, suffix='') -> dict[str, str]:
        """
        Returns the sensor keys of the simulator.
        
        Args: 
            suffix: String to be added after the sensor name, default ''.
        """
        sensor_keys = {}
        for sensor in self.sensors:
            sensor_keys[sensor.name] = sensor.name + suffix
        return sensor_keys