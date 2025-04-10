import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .base_material_estimator import BaseVSFMaterialEstimator
from ..core.neural_vsf import NeuralVSF
from ..sim.quasistatic_sim import QuasistaticVSFSimulator
from ..sim.sim_state_cache import SimStateCache
from ..dataset import BaseDataset,DatasetConfig
from ..sensor import BaseCalibrator
from ..utils.data_utils import convert_to_tensor, remap_dict_in_seq
from dataclasses import dataclass
from typing import Dict,List,Union

@dataclass
class NeuralVSFEstimatorConfig:
    """
    Configuration for NeuralVSFEstimator.
    """
    batch_size : int = 1
    """Batch size for the optimizer.  Currently only supports batch size of 1."""
    lr: float = 1e-3
    """Learning rate for the optimizer."""
    lr_decay_factor : float = 0.1
    """Factor by which to reduce the learning rate."""
    lr_decay_patience : int = 0
    """Number of epochs with no improvement after which learning rate will be reduced."""
    lr_decay_min : float = 1e-6
    regularizer_samples: int = 1000
    """Number of points to sample for the regularization term.  This is used to regularize
    the stiffness in unobserved regions towards zero."""
    regularizer_scale: float = 1e-8
    """The scale of the regularization term.  This is used to regularize the stiffness
    in unobserved regions towards zero. """
    max_epochs : int = 100
    """Maximum number of epochs to train for."""
    down_sample_rate: int = 1
    """The downsample rate for the dataset.  This is used to reduce the number of
    samples used for training.  This downsample rate is used to select every nth 
    frame for training.
    """


class NeuralVSFEstimator(BaseVSFMaterialEstimator):
    """
    Neural VSF stiffness estimator.  Works in online or batch mode.
    """
    def __init__(self, config : NeuralVSFEstimatorConfig):
        self.config = config
        self.vsf = None
        self.optimizer = None
        self.scheduler = None  
        assert config.batch_size == 1, "Batch size > 1 not supported yet"
    
    def online_init(self, sim : QuasistaticVSFSimulator, vsf : NeuralVSF):
        """Note: for best results, call vsf.to('cuda') to train on GPU before calling this."""
        assert isinstance(sim, QuasistaticVSFSimulator)
        assert isinstance(vsf, NeuralVSF)
        self.vsf = vsf
        self.optimizer = torch.optim.Adam(vsf.vsfNetwork.get_params(self.config.lr))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.lr_decay_factor, patience=self.config.lr_decay_patience, min_lr=self.config.lr_decay_min)
        self.vsf.vsfNetwork.train()

    def online_update(self, sim : QuasistaticVSFSimulator, dt, observations : Dict[str,np.ndarray]) -> float:
        for sensor in sim.sensors:
            assert sensor.name in observations
        for sensor_name in observations:
            assert sim.get_sensor(sensor_name) is not None
        LOSS_SCALE = 1.0
        vsf = self.vsf
        aabb = vsf.vsfNetwork.aabb
        prediction =  {}
        for sensor in sim.sensors:
            prediction[sensor.name] = sensor.predict_torch(sim.state())

        # regularization term, enforce low stiffness in unobserved region
        vsf_samples = torch.rand(self.config.regularizer_samples, 3).to(vsf.device) * (aabb[1] - aabb[0]) + aabb[0]
        stiffness = vsf.getStiffness(vsf_samples)
        
        loss = 0
        for sensor_name, o in observations.items():
            p = prediction[sensor_name]
            o = torch.tensor(o,dtype=p.dtype,device=p.device)
            loss = F.mse_loss(p.reshape(-1), o.reshape(-1))
        loss += torch.abs(stiffness).mean().to(loss.device) * self.config.regularizer_scale

        # optimize VSF model
        self.optimizer.zero_grad()
        # LOSS_SCALE to prevent gradient underflow
        (loss * LOSS_SCALE).backward()
        # TODO: try different learning rate scheduler
        # self.scheduler.step(loss) # loss can be unbalanced for each simulation time step, currently not adjusting learning rate based on loss
        self.optimizer.step()

        return loss.item() # For logging, decide if we want to return loss.

    def online_reset(self, sim: QuasistaticVSFSimulator):
        pass
    
    def online_finalize(self):
        self.vsf.vsfNetwork.eval()

    def batch_estimate(self,
                       sim : QuasistaticVSFSimulator,
                       vsf : NeuralVSF,
                       dataset: BaseDataset,
                       dataset_metadata : DatasetConfig,
                       calibrators : Dict[str,BaseCalibrator] = None,
                       dt = 0.1,
                       verbose:bool=False) -> torch.Tensor:
        """
        Solver that optimizes the NeuralVSF model to match the observations
        from a dataset.

        Note: for best results, call vsf.to('cuda') to train on GPU before calling this.

        TODO: the batch_estimate function currently evaluates sequences sequentially. To
        do a better job of shuffling training, we would need to store the internal state
        of the neural VSF simulator for each frame. This is not currently supported.
        """        
        self.online_init(sim,vsf)
        
        epoch_iterator = range(self.config.max_epochs)
        pbar = tqdm(epoch_iterator) if verbose else epoch_iterator

        for epoch in pbar:
            
            #pick a random sequence
            seq = dataset[torch.randint(0,len(dataset),(1,)).item()]
            
            # collect controls and observations in the sequence
            # if control_keys/sensor_keys are not provided, default to use object/sensor names for default keys
            control_keys = dataset_metadata.control_keys if len(dataset_metadata.control_keys) != 0 else sim.get_control_keys()
            sensor_keys = dataset_metadata.sensor_keys if len(dataset_metadata.control_keys) != 0 else sim.get_sensor_keys()
            control_seq, observation_seq = remap_dict_in_seq(seq, control_keys, sensor_keys)

            sim.reset()
            self.online_reset(sim)
            
            #calibrate the sensors from the sequences
            ncalibrate = 0
            if calibrators is not None and calibrators != {}:
                for k,v in calibrators.items():
                    sensor = sim.get_sensor(k)
                    if sensor is None: raise ValueError(f"Sensor {k} not found in simulator")
                    n = v.calibrate(sensor, sim, control_seq, observation_seq)
                    ncalibrate = max(n,ncalibrate)

                    if verbose:
                        print(f'Calibrated sensor {k} using {n} time steps')

            for i in range(ncalibrate, len(seq), self.config.down_sample_rate):
                # step simulation to get current sensor prediction
                if verbose:
                    print(f"Step {i-ncalibrate}/{len(seq)-ncalibrate}")
                    free, total = torch.cuda.mem_get_info()
                    print(f"Available CUDA memory: {free / 1024**3:.2f} GB, total: {total / 1024**3:.2f} GB")
                controls = control_seq[i]
                observations = observation_seq[i]
                sim.step(controls,dt)
            
                self.online_update(sim, dt, observations)

        self.online_finalize()