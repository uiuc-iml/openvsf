"""
Recursive solver that estimates PointVSF using
data feed into the system frame-by-frame.
"""
import numpy as np
import torch
import time
from pathlib import Path
import json

from ..core.base_vsf import BaseVSF
from ..core.point_vsf import PointVSF
from ..sim.sim_state_cache import SimStateCache
from ..dataset import BaseDataset,DatasetConfig
from ..sim.quasistatic_sim import QuasistaticVSFSimulator
from ..sensor.base_sensor import ContactState, SimState
from ..sensor.base_calibrator import BaseCalibrator
from ..prior.prior_factory import BaseVSFPriorFactory
from ..prior.structured_prior_factory import BaseVSFStructuredPriorFactory
from .base_material_estimator import BaseVSFMaterialEstimator
from .recursive_optimizer import SGDEstimator,DiagonalEKF,DenseEKF,ObservationLinearization
from .quad_prog_optimizer import QuadProgOptimizer
from ..prior.distribution import DiagGaussianDistribution,FullGaussianDistribution
from ..utils.perf import PerfRecorder, DummyRecorder

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

@dataclass
class PointVSFEstimatorConfig:
    """Defines parameters for a PointVSFEstimator."""
    optimizer : str = 'quad_prog'
    """estimator type: 'quad_prog', 'sgd', 'diag_ekf', 'dense_ekf'"""
    batch_size : int = 100
    """batch size for sgd optimizer"""
    num_replay_update : int = 40
    """how many prior replay updates to do during the update"""
    max_buffer_length : int = 2000
    """max replay buffer length for a recursive estimator"""
    down_sample_rate : int = 1
    """down samples input observations at this rate to speed up estimation"""
        

class PointVSFEstimator(BaseVSFMaterialEstimator):
    """
    A class that estimates a PointVSF stiffness field.

    It can accept a prior factory and a meta-prior to help guide the estimation.
    
    Very important!!! This class has VSF object and simulators, but they ARE NOT attached 
    to the estimator, it is only a pointer to the recent VSF. When you would to estimate 
    a new VSF, you need to call the online_init function to refresh the pointer to the VSF. 
    Batch estimation will refresh the VSF pointer automatically when using the batch_estimate 
    function.
    
    Attributes:
        config (PointVSFEstimatorConfig): Configuration for the estimator.
        prior_factory (BaseVSFPriorFactory): Prior factory for the estimator.
        struct_prior_factory (BaseVSFStructuredPriorFactory): Structured prior factory for the estimator.
        vsf (PointVSF): Temporary reference to the VSF object being estimated.
        vsf_sim (QuasistaticVSFSimulator): Temporary reference to the simulator object used for estimation.
        estimator: The optimizer used for estimation.
    """
    def __init__(self, config: PointVSFEstimatorConfig, 
                 prior : BaseVSFPriorFactory, meta_prior : BaseVSFStructuredPriorFactory = None):
        self.config = config
        self.prior = prior
        self.meta_prior = meta_prior
        self.vsf = None
        self.vsf_sim = None
        self.estimator = None
    
    def prior_predict(self, vsf : PointVSF) -> DiagGaussianDistribution:
        """
        Predicts the prior stiffness distribution for the given VSF.
        """
        K_dist = self.prior.predict(vsf)
        if self.meta_prior is not None:
            phi_basis = self.meta_prior.phi_basis(vsf)
            phi_prior = self.meta_prior.phi_prior(vsf)
            if isinstance(phi_prior,FullGaussianDistribution):
                K_meta_prior = FullGaussianDistribution(phi_basis @ phi_prior.mu, phi_basis @ phi_prior.var @ phi_basis.T)
            elif isinstance(phi_prior,DiagGaussianDistribution):
                K_meta_prior = DiagGaussianDistribution(phi_basis @ phi_prior.mu, (phi_basis @ torch.diag(phi_prior.var) @ phi_basis.T).diagonal())
            else:
                raise ValueError("Unsupported prior type")

            return K_dist + K_meta_prior
        return K_dist

    def setup_optimizer(self, vsf : PointVSF, initialize_vsf_features = True, verbose=True):
        """
        Setup the recursive/batched optimizer for the given VSF.
        """
        assert self.config.optimizer in ['quad_prog','sgd','diag_ekf','dense_ekf'], f"unsupported optimizer: {self.config.optimizer}"
        if vsf.axis_mode != 'isotropic':
            raise NotImplementedError("Only isotropic axis mode is supported for now")
        num_material_params = vsf.stiffness.reshape(-1).size(0)
        # disable gradient during the prior prediction
        with torch.no_grad():
            stiffness_dist = self.prior_predict(vsf)
            mu, std = stiffness_dist.param_mean(), stiffness_dist.param_std()
        assert mu.size(0) == num_material_params
        assert std.size(0) == num_material_params
        assert mu.shape == (vsf.rest_points.size(0), ), "can only handle isotropic stiffness for now"
        assert std.shape == (vsf.rest_points.size(0), ), "can only handle isotropic stiffness for now"
        if initialize_vsf_features:
            vsf.stiffness = mu
            vsf.features['K_std'] = std
            if 'N_obs' not in vsf.features:
                vsf.features['N_obs'] = torch.zeros(self.vsf.num_points, dtype=int).to(self.vsf.stiffness.device)

        latent_mu = 0.0
        latent_std = 0.0
        latent_basis = None
        if self.meta_prior is not None:
            phi_prior = self.meta_prior.phi_prior(vsf)
            phi_matrix = self.meta_prior.phi_basis(vsf)
            latent_mu = phi_prior.param_mean()
            latent_std = phi_prior.param_std()
            latent_basis = phi_matrix
        
        # Initialize common parameters for the all optimizer
        common_params = { 'max_dim': num_material_params, 'x_mu': mu, 'x_var': std**2,
                          'latent_mu': latent_mu, 'latent_var': latent_std**2, 'latent_basis': latent_basis,
                          'max_buffer_len': self.config.max_buffer_length }
        if self.config.optimizer == 'sgd':
            estimator = SGDEstimator(**common_params, non_negative=True, batch_size=self.config.batch_size)
        elif self.config.optimizer == 'diag_ekf':
            estimator = DiagonalEKF(**common_params, non_negative=True, num_replay_update=self.config.num_replay_update)
        elif self.config.optimizer == 'dense_ekf':
            estimator = DenseEKF(**common_params, non_negative=True, num_replay_update=self.config.num_replay_update)
        elif self.config.optimizer == 'quad_prog': 
            estimator = QuadProgOptimizer(**common_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        return estimator

    def online_init(self, sim : QuasistaticVSFSimulator, vsf : PointVSF):
        assert isinstance(vsf, PointVSF)
        assert isinstance(sim, QuasistaticVSFSimulator)
        matching_vsfs = [k for (k,v) in sim.vsf_objects.items() if v.vsf_model is vsf]
        if len(matching_vsfs) == 0:
            raise ValueError("VSF object is not present in simulators, available objects: {}".format(','.join(sim.vsf_objects.keys())))
        vsf_name = matching_vsfs[0]
        self.vsf = vsf
        self.vsf_sim = sim

        if self.config.optimizer == 'quad_prog':
            raise NotImplementedError("Online initialization is not supported for quad_prog estimator")
        self.estimator = self.setup_optimizer(vsf) 
        self.online_reset(sim)

    def online_reset(self, sim : QuasistaticVSFSimulator):
        """
        Clear the estimator state to prepare for next online estimation.
        
        Different from online_init that fully reset the estimator, this function
        only clears the observation buffer.
        """
        assert isinstance(sim, QuasistaticVSFSimulator)
        self.estimator.clear_obs_buffer()
    
    def online_update(self, sim : QuasistaticVSFSimulator, dt:float, measurements:Dict[str,np.ndarray],
                      perfer: PerfRecorder=DummyRecorder(), verbose=False):
        """
        Update point VSF stiffness given updated simulator state and sensor measurements.
        
        NOTE: this function assumes controls are externally applied to sim by the user. 
        """
        assert isinstance(sim, QuasistaticVSFSimulator)
        assert isinstance(measurements, dict)
        assert len(measurements) == 1, "only support single sensor update"
        sensor_name = list(measurements.keys())[0]
                
        perfer.start('linearize_observation')
        obs = self.linearize_observation(self.vsf_sim.state(), sensor_name)
        perfer.stop('linearize_observation')
        
        if verbose:
            print('number of observed points:', obs.state_indices.shape[0])
        
        if obs is not None:
            perfer.start('update_estimation')
            measurements_tensor = torch.from_numpy(measurements[sensor_name]).to(obs.matrix.device)
            self.estimator.add_observation(obs, measurements_tensor)
            self.estimator.update_estimation(verbose=False)
            perfer.stop('update_estimation')

            if obs.state_indices is not None:
                self.vsf.features['N_obs'][obs.state_indices] += 1
            else:
                self.vsf.features['N_obs'] += 1

    def online_finalize(self, vsf : PointVSF):
        """
        Finalizes the estimation. Run post-processing steps to update the
        vsf's stiffness value, 'K_std' and 'N_obs' features.
        """
        assert isinstance(vsf, PointVSF)
        self.estimator.finalize_estimation()
        vsf.stiffness[:] = self.estimator.get_mean()
        vsf.features['K_std'][:] = self.estimator.get_var()

    def batch_estimate(self, sim,
                       vsf : PointVSF,
                       dataset : BaseDataset,
                       dataset_metadata : DatasetConfig,
                       calibrators : Dict[str,BaseCalibrator]=None,
                       dt = 0.1,
                       sim_out_dir: str = None):
        """
        Solves the VSF stiffness using from raw control and observation data.
        The solving procedure has the following steps:
        1. Simulate the sensor state using the simulator, caching the simulated states;
        this dataset can be optionally saved on the disk to avoid re-simulation.
        2. Solve VSF stiffness given the simulated sensor states.
        """
        sim_cache = SimStateCache(sim)
        if sim_out_dir is not None and Path(sim_out_dir).exists():
            sim_cache.load(sim_out_dir)
        else:
            if sim_out_dir is not None:
                print("Generating simulation cache and saving to",sim_out_dir)
            else:
                print("Generating simulation cache")
            sim_cache.generate(dataset, dataset_metadata, calibrators, dt)
            if sim_out_dir is not None:
                sim_cache.save(sim_out_dir)
        self.batch_estimate_sim_dataset(sim, vsf, sim_cache, verbose=True)

    def batch_estimate_sim_dataset(self, sim : QuasistaticVSFSimulator, vsf : PointVSF, sim_dataset:SimStateCache, verbose=False):
        """
        A utility function that solves VSF stiffness given simulated sensor states.
        This function is not expected to be called directly by the user, which should be used by 
        the solve function only
        
        Args:
        - sim_dataset: A cache of the simulation state and sensor states.
        """
        assert len(sim_dataset) > 0, 'ERROR: empty simulation dataset'
        assert isinstance(vsf, PointVSF), 'ERROR: vsf is not a PointVSF'
        assert isinstance(sim, QuasistaticVSFSimulator), 'ERROR: sim is not a QuasistaticVSFSimulator'
        try:
            vsf_name = [k for (k,v) in sim.vsf_objects.items() if v.vsf_model is vsf][0]
        except:
            all_vsf_names = ','.join([k for k in sim.vsf_objects.keys()])
            raise ValueError(f"VSF object is not present in simulators, available objects: {all_vsf_names}")

        self.vsf_sim = sim
        self.vsf = vsf
        linearized_models = self.linearize_dataset(sim_dataset, flatten=True, verbose=verbose)

        estimator = self.setup_optimizer(vsf) 
        
        # down-sample the input data to speed up the estimation
        if self.config.down_sample_rate > 1:
            linearized_models = linearized_models[::self.config.down_sample_rate]
            if verbose:
                print('Batch estimation down-sampling with rate',self.config.down_sample_rate, 
                      'to',len(linearized_models),'frames')
        
        for model,meas in linearized_models:
            estimator.add_observation(model, meas)
        if self.config.optimizer != 'quad_prog':
            estimator.update_estimation()
        estimator.finalize_estimation()
        vsf.stiffness[:] = estimator.get_mean()
        vsf.features['K_std'][:] = estimator.get_var()
        if 'N_obs' not in vsf.features:
            vsf.features['N_obs'] = torch.zeros(vsf.num_points, dtype=int)
        for model,meas in linearized_models:
            if model.state_indices is not None:
                vsf.features['N_obs'][model.state_indices] += 1
            else:
                vsf.features['N_obs'] += 1 
        

    def linearize_dataset(self, sim_dataset: SimStateCache, flatten=True, 
                          verbose=False) -> List[Tuple[ObservationLinearization, np.ndarray]]:
        """
        Reduces multiple sequences of point VSF simulation data to linear observation data.

        Args:
            sim_dataset (SimStateCache): 
                The cache containing multiple sequences of simulation states.
            flatten (bool, optional): 
                If `True`, all sequences are concatenated into a single sequence 
                of `(observation_model, obs)` tuples, and all `None` models are ignored.
                Defaults to `True`.
            verbose (bool, optional): 
                If `True`, prints additional debugging information. Defaults to `False`.

        Returns:
            list: A list of sequences. Each frame in a sequence is a dictionary mapping 
            sensor names to:
            
            - `(None, obs)`: If the observation model is ignored.
            - `(linear_observation_model, obs)`: If the observation model is included.

            If `flatten=True`, all sequences are concatenated into a single sequence 
            of `(observation_model, obs)` tuples, and all `None` models are ignored.
        """
        res = []
        for seq in sim_dataset:
            lin_sequence = []
            for sim_state in seq:
                self.vsf_sim.load_state_dict(sim_state)
                sensors_state = sim_state['sensors']
                observation_dict = {}
                for sensor in self.vsf_sim.sensors:
                    sensor.set_calibration(sensors_state[sensor.name].get('calibration', None))
                    meas = sensors_state[sensor.name].get('observation', None)
                    obs_model = self.linearize_observation(self.vsf_sim.state(), sensor.name)
                    observation_dict[sensor.name] = (obs_model, meas)
                lin_sequence.append(observation_dict)
            res.append(lin_sequence)
        if verbose:
            print('Number of observations:', len(res))
        if flatten:
            flattened = []
            for seq in res:
                for frame in seq:
                    for sensor,value in frame.items():
                        if value[0] is not None and value[1] is not None:
                            flattened.append(value)
            return flattened
        return res

    def linearize_observation(self, sim_state: SimState, sensor_name:str) -> Optional[ObservationLinearization]:
        """For the given sensor, linearize the observation and return
        the observation matrix and the observation index.
        
        Returns:
            ObservationLinearization: the linearized model as a function of the object
            stiffness state variable.
        """
        sensor = self.vsf_sim.get_sensor(sensor_name)
        sensor_dim = len(sensor.measurement_names())
        
        vsf_name = [k for (k,v) in self.vsf_sim.vsf_objects.items() if v.vsf_model is self.vsf][0]
        for sensor in self.vsf_sim.sensors:
            if sensor.name == sensor_name: break
        
        #go through bodies that are contacted by the sensor
        contact_jacobian_dict = sensor.measurement_force_jacobian(sim_state)
        vsf_idxs = []
        curr_pts = []
        vsf_contact_jacobians = []
        for (sensor_body, other_body),J in contact_jacobian_dict.items():
            if other_body == vsf_name:
                contact_state = sim_state.contacts[(vsf_name,sensor_body)]
                vsf_idxs.append(contact_state.elems1)
                curr_pts.append(contact_state.points)
                assert len(contact_state.elems1) == len(contact_state.points)
                assert J.shape[1] == len(contact_state.points)
                vsf_contact_jacobians.append(J)
        if len(vsf_idxs) == 0:
            return None
        
        vsf_idxs = torch.concat(vsf_idxs)
        curr_pts = torch.concat(curr_pts)
        contact_jacobian = torch.concat(vsf_contact_jacobians,dim=1)                
        
        assert isinstance(contact_jacobian,torch.Tensor)
        sensor_dim = contact_jacobian.size(0)
        assert contact_jacobian.size(1) == len(curr_pts)
        assert contact_jacobian.size(1) == len(vsf_idxs)

        #get derivative of sensor measurement with respect to vsf material parameters
        #d sensor / d mat = dsensor / dforces * dforces / d mat
        #d forces / d mat = block_diag(-(curr_pts - rest_pts)) for isotropic
        #d forces / d mat = block_diag(-diag(curr_pts - rest_pts)) for axis aligned
        # d sensor_i / d_mat_j = sum_l=1^3 J[i,j,l] * (-curr_pts[j]-rest_pts[j])[l]   for axis aligned
        # d sensor_i / d_mat_j = J[i,j//3,j%3] * (-curr_pts[j//3]-rest_pts[j//3])[j%3]   for isotropic
        rest_pts = self.vsf.rest_points[vsf_idxs]
        if self.vsf.axis_mode == 'isotropic':
            obs_idx = vsf_idxs
            neg_dr = -(curr_pts - rest_pts)
            obs_W = (contact_jacobian * neg_dr[None]).sum(dim=2)
            assert obs_W.ndim == 2
            assert obs_W.size(0) == sensor_dim
            assert obs_W.size(1) == len(vsf_idxs)
        elif self.vsf.axis_mode == 'axis_aligned':
            obs_idx = torch.stack([vsf_idxs*3, vsf_idxs*3+1, vsf_idxs*3+2], dim=1).reshape(-1)
            neg_dr = -(curr_pts - rest_pts)
            obs_W = contact_jacobian.reshape(sensor_dim, -1) * neg_dr[None,:]
            assert obs_W.ndim == 2
            assert obs_W.size(0) == sensor_dim
            assert obs_W.size(1) == len(vsf_idxs)*3
        else:
            raise NotImplementedError(f"Symmetry type {self.vsf.axis_mode} not implemented")
        
        bias = sensor.predict(SimState(sim_state.body_transforms, sim_state.body_states))
        var = sensor.measurement_errors()
        if var is None:
            var = 1.0
        else:
            var = torch.tensor(var,device=obs_W.device,dtype=obs_W.dtype)**2
        return ObservationLinearization(matrix = obs_W, var=var, bias = bias, state_indices = obs_idx)

    def _eval_predict_debug(self, obs_info_dict: dict[str, np.ndarray], verbose=False) -> list[Dict[str,np.ndarray]]:
        """
        A utility function that directly evaluate the tactile prediction accuracy given the
        obs_info_dict, which are summarized linear observation matrix and vector.
        
        Args:
        - obs_info_dict: A result from the linearize_dataset function.
        - verbose (bool): A flag to print the evaluation process.

        Returns:
            A list of # sequences dictionaries, each containing the tactile de-biased
            observation and prediction for the sequence.
        """

        start_time = time.time()
        all_observed_mat_idx = obs_info_dict['all_merge_mat_idx']
        seq_lengths = obs_info_dict['seq_lengths']
        n_obs_total = sum(seq_lengths)
        obs_err_info = [{} for _ in range(len(seq_lengths))]

        for sensor in self.vsf_sim.sensors:
            sensor_name = sensor.name

            num_mat_points = all_observed_mat_idx.shape[0]
            obs_mat = obs_info_dict[f'{sensor_name}_obs_mat']
            obs_vec = obs_info_dict[f'{sensor_name}_obs_vec']
            assert obs_mat.shape == (obs_vec.shape[0],obs_vec.shape[1],num_mat_points)
            assert obs_vec.shape[0] == n_obs_total
            sensor_dim = obs_vec.shape[1]
            obs_mat = obs_mat.reshape(num_mat_points, -1)
            obs_vec = obs_vec.reshape(-1, 1)

            if verbose:
                print('obs_dim:', num_mat_points)
                print('obs_mat shape:', obs_mat.shape)
                print('obs_vec shape:', obs_vec.shape)

            # NOTE: get_K_est handles add/replace basis functions
            K_mu = self.estimator.get_mean(idx=all_observed_mat_idx).cpu().numpy()
            hat_vec = obs_mat.T @ K_mu.reshape(-1, 1)
            print('eval time:', time.time() - start_time)

            hat_tau_seq = hat_vec.reshape(-1, sensor_dim)
            obs_tau_seq = obs_vec.reshape(-1, sensor_dim)

            split_size_lst = np.cumsum(seq_lengths[:-1])
            split_hat_vec = np.split(hat_tau_seq, split_size_lst, axis=0)
            split_obs_vec = np.split(obs_tau_seq, split_size_lst, axis=0)

            for seq_idx, (hat_tau, obs_tau) in enumerate(zip(split_hat_vec, split_obs_vec)):
                
                obs_err_info[seq_idx].update({ f'{sensor_name}_sim': hat_tau, 
                                          f'{sensor_name}_obs': obs_tau })

        return obs_err_info

    def _eval_sim_dataset(self, sim_dataset : SimStateCache, verbose=False) -> List[np.ndarray]:
        """
        Evaluate the tactile observations prediction accuracy given the simulated sensor states.
        
        Args:
        - dataset: A simulated state dataset containing the simulated sensor states.
        - verbose (bool): A flag to print the evaluation process.
        """
        
        linearized_models = self.linearize_dataset(sim_dataset, verbose=verbose) 
        predictions = {} 
        measurements = {}
        
        for sensor in self.vsf_sim.sensors:
            start_time = time.time()
            predictions[sensor.name] = []
            measurements[sensor.name] = []
            for i,seq in enumerate(linearized_models):
                if verbose:
                    print("Evaluating sensor",sensor.name,'sequence:', i)
                pred_i = []
                meas_i = []
                for frame in seq:
                    obs_model = frame[sensor.name][0]
                    if obs_model is None:
                        continue
                    obs_vec = frame[sensor.name][1]
                    pred = obs_model.predict(self.vsf.stiffness.view(-1))
                    pred_i.append(pred)
                    meas_i.append(obs_vec)
                predictions[sensor.name].append(pred)
                measurements[sensor.name].append(obs_vec)
            print('Evaluation time:', time.time() - start_time)
        return predictions,measurements
    
