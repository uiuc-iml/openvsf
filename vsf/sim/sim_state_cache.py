from __future__ import annotations
import numpy as np
import torch
import json

from .quasistatic_sim import QuasistaticVSFSimulator
from ..dataset import BaseDataset,DatasetConfig
from ..utils.perf import PerfRecorder, DummyRecorder
from ..utils.data_utils import remap_dict_in_seq
from ..sensor.base_calibrator import BaseCalibrator
from ..sensor.base_sensor import ContactState, SimState

from typing import List, Dict, Tuple, Union

class SimStateCache(BaseDataset):
    """Caches simulation state sequences for a VSF simulator.
    
    The result of the cache is a list of sequences, where each sequence is a
    list of states.  Each state is a dictionary containing the state of the
    simulator at that time step.
    
    A state dictionary will contain the following keys:

    - `control` : the control inputs at that time step
    - `robot_configs` : the robot configuration at that time step
    - `geom_transforms`: the geometry transforms at that time step
    - `vsf_objects` : the VSF object states at that time step
    - `contact_state` : the contact state of the simulator
    - `sensors`: a dictionary of sensor states at that time step
    - `t` : the time of the state
    - `dt` : the time step of the state

    TODO: determine a convention for the format.  Should they be numpy arrays or tensors?
    """
    def __init__(self, sim : QuasistaticVSFSimulator, save_sensors : bool = True):
        self.sim = sim
        self.save_sensors = save_sensors
        self.sequences = []
    
    def save(self, filename : str):
        """Saves the cache to a file or folder."""
        import os
        if os.path.isdir(filename):
            filename = os.path.join(filename,'sim_cache.json')
        with open(filename, 'w') as f:
            json.dump(self.sequences, f, indent=2)
    
    def load(self, filename : str):
        """Loads the cache from a file or folder."""
        import os
        if os.path.isdir(filename):
            filename = os.path.join(filename,'sim_cache.json')
        with open(filename, 'r') as f:
            self.sequences = json.load(f)

    def get_sequence(self, idx : int):
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)
    
    def subsample(self, seq_idxs : Union[int,List[int]], time_idxs : Union[int,List[int]]) -> SimStateCache:
        """Subsamples the cache to only include the given sequences and time
        indices.

        If integers are provided, they are treated as subsampling rates.
        """
        if isinstance(seq_idxs,int):
            seq_idxs = list(range(0,len(self.sequences),seq_idxs))
        res = SimStateCache(self.sim,self.save_sensors)
        sequences = [self.sequences[i] for i in seq_idxs]
        for seq in sequences:
            if isinstance(time_idxs,int):
                time_idxs = list(range(0,len(sequences[0]),time_idxs))
            seq = [seq[i] for i in time_idxs if i < len(seq)]
            res.sequences.append(seq)
        return res

    def trajectory(self, seqidx : int, path : List[str]) -> Union[torch.Tensor,np.ndarray]:
        """Returns a trajectory over an item identified at the
        given path."""
        seq = self.sequences[seqidx]
        items = []
        for frame in seq:
            item = frame
            for p in path:
                item = item[p]
            items.append(item)
        if any(isinstance(x,torch.Tensor) for x in items):
            return torch.stack(items)
        else:
            return np.stack(items)

    def robot_trajectory(self, seqidx : int, name : str = None) -> np.ndarray:
        """Returns a trajectory over the given robot."""
        if name is None:
            name = self.sequences[idx][0]['robot_config'].keys()[0]
        return self.trajectory(seqidx,['robot_config',name])
    
    def rigid_object_trajectory(self, seqidx : int, name : str = None) -> np.ndarray:
        """Returns a trajectory over the given rigid object."""
        if name is None:
            name = self.sequences[idx][0]['geom_transforms'].keys()[0]
        return self.trajectory(seqidx,['geom_transforms',name])

    def generate(self, dataset : BaseDataset,
                       dataset_metadata : DatasetConfig,
                       calibrators : Dict[str,BaseCalibrator]=None,
                       dt = 0.1):
        """Estimates a VSF's material parameters given a complete dataset.

        The simulator is usually a QuasistaticVSFSimulator.  It should contain
        all the sensors that are referenced in the dataset_metadata's 
        `sensor_keys` attribute.

        If calibrators are provided, they should be run at the start of each 
        trial to calibrate the sensors. 
        
        The VSF's stiffness parameters will be updated to the estimated values.

        The default implementation runs online estimation for all sequences in
        the dataset.
        """
        self.sequences = []
        for seqno in range(len(dataset)):
            #loop through each sequence in the dataset
            self.generate_append(dataset[seqno], dataset_metadata, calibrators, dt)
    
    def generate_append(self, seq : List[Dict[str,np.ndarray]],
                        dataset_metadata : DatasetConfig,
                        calibrators : Dict[str,BaseCalibrator]=None,
                        dt = 0.1) -> List[Dict,str,np.ndarray]:
        """Creates a new sequence in the cache corresponding to a
        dataset sequence."""
    
        control_keys = dataset_metadata.control_keys if len(dataset_metadata.control_keys) != 0 else self.sim.get_control_keys()
        sensor_keys = dataset_metadata.sensor_keys if len(dataset_metadata.sensor_keys) != 0 else self.sim.get_sensor_keys()

        control_seq, sensor_seq = remap_dict_in_seq(seq, control_keys, sensor_keys)
    
        self.sim.reset()
        n = 0
        if calibrators is not None:
            for k,calibrator in calibrators.items():
                sensor = self.sim.get_sensor(k)
                assert sensor is not None
                nk = calibrator.calibrate(sensor, self.sim, control_seq, sensor_seq)
                n = max(nk,n)
        state_seq = []
        t = n*dt
        for frameno in range(n,len(seq)):
            self.sim.step(control_seq[frameno],dt)
            state_seq.append(self.sim_state(t,dt, control_seq[frameno], sensor_seq[frameno]))
            t += dt
        self.sequences.append(state_seq)
        return state_seq

    def sim_state(self, t, dt, controls : dict[str,np.ndarray], observations: dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        res = self.sim.state_dict()
        res['t'] = t
        res['dt'] = dt
        res['control'] = controls
        if self.save_sensors:
            sensor_state = {}
            for s in self.sim.sensors:
                sensor_state[s.name] = {'calibration':s.get_calibration(),'state':s.get_internal_state()}
                sensor_state[s.name]['prediction'] = s.predict(self.sim.state())
                sensor_state[s.name]['observation'] = torch.from_numpy(observations[s.name]).to(sensor_state[s.name]['prediction'].device)
                sensor_state[s.name]['jacobian'] = s.measurement_force_jacobian(self.sim.state())
            res['sensors'] = sensor_state
        return res

    def load_sim_state(self, sim_state : Dict[str,np.ndarray]):
        """Restores the simulator state from a saved state."""
        self.sim.load_state_dict(sim_state)
        if 'sensors' in sim_state:
            for sname,sdata in sim_state["sensors"].items():
                sensor = self.sim.get_sensor(sname)
                sensor.set_calibration(sdata['calibration'])
                sensor.set_internal_state(sdata['state'])
