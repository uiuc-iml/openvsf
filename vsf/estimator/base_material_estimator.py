from abc import ABC,abstractmethod
from ..dataset import BaseDataset, DatasetConfig
from ..core.base_vsf import BaseVSF
from ..sensor import BaseSensor, BaseCalibrator
from typing import Callable,Dict,List,Tuple,Union,Optional
import numpy as np


class BaseVSFMaterialEstimator(ABC):
    """
    Base class for an estimator of a VSF's material parameters. 

    Estimators can be run in batch mode or in online mode.  In batch mode, the
    estimator is given a complete dataset and estimates the material parameters
    of the VSF.  In online mode, the estimator is given a simulator and a
    sequence of control inputs and sensor observations, and updates the material
    parameters of the VSF in real time.

    Online usage:
    
        vsf = make_empty_vsf()
        sim = QuasistaticVSFSimulator(world, [sensor])
        sim.add_deformable(vsf)

        est = XEstimator(...)
        est.online_init(sim,vsf)
        est.online_reset(sim)
        for t in range(T):
            #read control and sensor measurements for your system
            control = {ROBOTNAME:get_robot_control(t)}
            measurements = {sensor.name:get_sensor_measurements(t)}
            #advance simulator and estimator
            sim.step(control,dt)
            est.online_update(sim,dt,measurements)
        est.online_finalize(vsf)
        vsf.save('est_vsf.npy')

    """
    def __init__(self):
        pass

    def batch_estimate(self, sim,
                       vsf : BaseVSF,
                       dataset : BaseDataset,
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
        #reset parameters
        self.online_init(sim,vsf)
        for seqno in range(len(dataset)):
            #loop through each sequence in the dataset
            seq = dataset[seqno]
            control_seq = []
            sensor_seq = []
            for frame in seq:
                control_seq.append({k:frame[v] for k,v in dataset_metadata.control_keys.items()})
                sensor_seq.append({k:frame[v] for k,v in dataset_metadata.sensor_keys.items()})
        
            sim.reset()
            n = 0
            if calibrators is not None:
                for k,calibrator in calibrators.items():
                    sensor = sim.get_sensor(k)
                    assert sensor is not None
                    nk = calibrator.calibrate(sensor, sim, control_seq, sensor_seq)
                    n = max(nk,n)
            self.online_reset(sim)
            for frameno in range(n,len(seq)):
                sim.step(control_seq[frameno],dt)
                observations = {k:sensor_seq[frameno][k] for k in dataset_metadata.sensor_keys}
                self.online_update(sim,dt,observations)
            self.online_finalize(vsf)

    @abstractmethod
    def online_init(self, sim, vsf : BaseVSF):
        """Initializes online estimation version of this estimator.
        
        Any internal state and material priors should be instantiated here.

        Args:
            sim is the simulator that will be used for online estimation.
            vsf is the VSF that will be estimated.
        """
        raise NotImplementedError

    @abstractmethod
    def online_reset(self, sim):
        """Finishes a trial and resets the online estimation version
        of this estimator to a new trial.  The sim is assumed to have been
        reset.  (The material parameters of the VSF are kept the same.)
        """
        pass

    @abstractmethod
    def online_update(self, sim, dt:float, observations : dict):
        """Given a simulator advanced by dt to a new time step and the true
        observations received at that time step, perform an update of the
        material estimation. 

        The discrepancy between sim.measurements() and observation will
        be used to update the material parameters.
        """
        raise NotImplementedError

    def online_finalize(self, vsf : BaseVSF):
        """If the online estimator doesn't update the VSF in-place, this
        operation should be called to update the VSF's material parameters.
        """
        pass

    def online_state(self) -> dict:
        """Returns any internal state of the online estimator (for use with
        online_load_state).  Only needed if you want to save and restore
        estimator state."""
        return {}
    
    def online_load_state(self, state : dict):
        """Loads internal state of the online estimator (for use with
        online_state).  Only needed if you want to save and restore
        estimator state."""
        pass

