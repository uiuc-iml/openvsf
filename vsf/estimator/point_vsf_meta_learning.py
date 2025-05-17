import numpy as np
import torch
from torch import optim
import time
from typing import List, Dict, Tuple
import os

from .point_vsf_estimator import PointVSFEstimator,PointVSFEstimatorConfig
from ..core.point_vsf import PointVSF
from ..prior.prior_factory import BaseVSFPriorFactory, LearnableVSFPriorFactory
from ..prior.structured_prior_factory import BaseVSFStructuredPriorFactory, LearnableVSFPriorFactory
from ..dataset import BaseDataset, DatasetConfig
from ..sim.quasistatic_sim import QuasistaticVSFSimulator
from ..sim.sim_state_cache import SimStateCache
from ..utils.perf import PerfRecorder, DummyRecorder
from ..sensor.base_sensor import ContactState, SimState
from ..sensor.base_calibrator import BaseCalibrator
from .recursive_optimizer import ObservationLinearization


class PointVSFMetaLearning:
    """Meta-learning priors or structured priors for PointVSF models.

    This learner contains an estimator and a prior factory and an 
    optional structured prior factory.

    There are two ways to do the learning:  
    
        The first one learns directly from a set of datasets, which
    optimizes the true Bayesian objective of the meta-learning
    This is more accurate and robust, but requires more computational effort.
    
        The second is to learn the prior from estimated stiffnesses.  
    This doesn't optimize the correct Bayesian objective of the meta-learning
    problem, but is much faster to run.
    """

    def __init__(self, estimator : PointVSFEstimator,
                 prior_factory : LearnableVSFPriorFactory,
                 struct_prior_factory : BaseVSFStructuredPriorFactory = None):
        self.estimator = estimator
        self.prior_factory = prior_factory
        self.struct_prior_factory = struct_prior_factory
        
    def meta_learn_laplacian_approx(self, vsfs : List[PointVSF]):
        """Performs meta learning from a set of already estimated VSFs.

        The targets are the vsf stiffnesses.
        """
        if self.meta_prior is None:
            #standard meta learning
            self.prior.meta_learn(vsfs)
        else:
            #meta prior learning: stiffness ~ K + A * phi
            self.meta_prior.eval()
            vsf_features = []
            phi_bases, psi_bases = [], []
            phis = []
            for vsf in vsfs:
                vsf_features.append(self.prior.feature_tensor(vsf))
                A, S = self.meta_prior.phi_psi_bases(vsf)
                phi_bases.append(A)
                psi_bases.append(S)
                mu_hat = self.meta_prior.phi_prior(vsf).param_mean()
                phis.append(torch.zeros_like(mu_hat,device=mu_hat.device))

            #learn mprior, prior, and phis simultaneously
            raise NotImplementedError("TODO: meta learning")
            self.meta_prior.prior.train()
            self.prior.train()
            mprior = self.meta_prior.meta_prior()
            loss = 0
            for i,vsf in enumerate(vsfs):
                phi_prior = self.meta_prior.phi_prior(vsf)  #this performs a forward-mode evaluation with respect to the meta-prior
                prior = self.prior.prior.predict(vsf_features)
                mean = prior.param_mean() + phi_bases[i] @ phis[i]
                var = prior.param_std()**2 + (phi_bases[i] @ phis[i])**2
                obs_loss = torch.sum((vsf.stiffness() - mean)**2 / var)
                prior_loss = torch.sum(phi_prior.log_prob(phis[i]))
                loss += obs_loss + prior_loss

    def meta_learn_bayesian(self,
                            datasets : List[BaseDataset],
                            dataset_configs : List[DatasetConfig],
                            vsfs : List[PointVSF],
                            simulators : List[QuasistaticVSFSimulator],
                            calibrators : List[BaseCalibrator] = None,
                            dt = 0.1, sim_out_dir=None, 
                            epochs=100, lr=0.01, verbose=False):
        """Meta-learning by minimizing the true Bayesian objective. 
        The learning objective is the marginalized likelihood of tactile 
        observations given meta-prior parameters.
        
        This function implements the optimization process of Eq. (1) in CoRL2024 paper.
        
        The internal likelihood function is evaluated using the `prior_loss` function in this module.
        
        Args:
            datasets (List[BaseDataset]): List of datasets to learn from.
            dataset_configs (List[DatasetConfig]): List of dataset configurations.
            vsfs (List[PointVSF]): Instantiate PointVSF objects for each dataset, note 
                these VSFs can be empty with random stiffness, but should contain 
                informative features for the meta-learning.
            simulators (List[QuasistaticVSFSimulator]): List of simulators, each simulator
                simulates the corresponding VSF object.
            calibrators (List[BaseCalibrator], optional): List of calibrators. Defaults to None.
            dt (float, optional): Time step for the simulation. Defaults to 0.1.
            
            sim_out_dir (str, optional): Directory to save the simulation cache. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 0.01.
            verbose (bool, optional): If True, print detailed information. Defaults to False.
        """
        
        # meta prior learning: stiffness ~ K + A * phi
        
        tsr_params = {'dtype': self.prior_factory.dtype,
                      'device': self.prior_factory.device}
        
        self.prior_factory.train()
        if self.struct_prior_factory is not None:
            self.struct_prior_factory.train()
        
        # generate simulation cache for each dataset
        merged_obs_model_list = []
        merged_measurement_list = []
        touched_vsf_list = []
        for dataset, dataset_config, vsf, simulator in zip(datasets, dataset_configs, vsfs, simulators):
            
            if dataset_config is None:
                dataset_config = DatasetConfig(
                    control_keys= simulator.get_control_keys(),
                    sensor_keys= simulator.get_sensor_keys(),
                )
            sim_cache = SimStateCache(simulator)   
            
            if sim_out_dir is None:
                sim_cache.generate(dataset, dataset_config, calibrators, dt)
            else:
                raise NotImplementedError("TODO: save simulation cache to disk")            
            
            # attach simulator and vsf to the estimator
            self.estimator.online_init(simulator, vsf)
            
            # linearize the observation model
            obs_dict_list: List[Dict] = self.estimator.linearize_dataset(sim_cache, flatten=True, 
                                                                         verbose=verbose)
            
            # dowsample the observations
            obs_dict_list = obs_dict_list[::self.estimator.config.down_sample_rate]
    
            obs_model_list : List[ObservationLinearization] = []
            measurement_list : List[torch.Tensor] = []
            for obs_model, measurement in obs_dict_list:
                obs_model_list.append(obs_model)
                measurement_list.append(measurement)
        
            merged_obs_model = ObservationLinearization.merge(*obs_model_list)
            merged_measurement = torch.cat(measurement_list)
            touched_vsf = vsf[merged_obs_model.state_indices]
            
            merged_obs_model_list.append(merged_obs_model.to(**tsr_params))
            merged_measurement_list.append(merged_measurement.to(**tsr_params))
            touched_vsf_list.append(touched_vsf.to(**tsr_params))

        parameters = self.prior_factory.parameters()
        if self.struct_prior_factory is not None:
            parameters += self.struct_prior_factory.parameters()
        
        if verbose:
            print('cuda mem:', torch.cuda.memory_allocated()/(1024**2))
            print('cuda max mem:', torch.cuda.max_memory_allocated()/(1024**2))

        optimizer = optim.Adam(parameters, lr=lr)
        
        for epoch in range(epochs):
            
            loss_lst = []
            for merged_obs_model, merged_measurement, touched_vsf in zip(
                merged_obs_model_list, merged_measurement_list, touched_vsf_list):
                
                loss = self.prior_loss(merged_obs_model, merged_measurement, touched_vsf,
                                       sig_form='obs+opt', verbose=verbose)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss.item())

            if verbose:
                print(f"Epoch {epoch}: Loss = {np.mean(loss_lst)}")


    def prior_loss(self, obs_model: ObservationLinearization,
                   measurement: torch.Tensor, vsf: PointVSF,  
                   sig_form='obs+opt', verbose=False) -> torch.Tensor:
        """
        
        Evaluate the marginalized likelihood of the linearized observation model 
        given the meta-prior (prior factory) parameters. The loss is the negative 
        log-likelihood of the observations given the meta-prior parameters.
        
        This function implements Eq. (14) in CoRL2024 paper.
        
        NOTE: this function by default will keep the gradient
        
        Args:
            obs_model (ObservationLinearization): The observation model.
            measurement (torch.Tensor): The measurement vector.
            vsf (PointVSF): The PointVSF object.
            sig_form (str, optional): The form of the covariance matrix. 
                Can be 'uniform', 'obs', or 'obs+opt'. Defaults to 'obs+opt'.
            verbose (bool, optional): If True, print detailed information. Defaults to False.
        Returns:
            torch.Tensor: The loss value.
        """
        assert sig_form in ['uniform', 'obs', 'obs+opt']

        tsr_params = { 'dtype': self.prior_factory.dtype,
                       'device': self.prior_factory.device }

        # Prepare observation matrix and vector
        
        obs_idx = obs_model.state_indices
        obs_mat_tsr = obs_model.matrix
        obs_vec_tsr = measurement.reshape(-1, 1)
        
        obs_var = obs_model.var

        num_obs = obs_vec_tsr.shape[0]

        if verbose:
            print('obs_idx shape:', obs_idx.shape)
            print('obs_mat_tsr shape:', obs_mat_tsr.shape)
            print('obs_vec_tsr shape:', obs_vec_tsr.shape)

        # NOTE: debug linear regression
        # c2m_mat_hat = torch.linalg.lstsq(obs_mat_tsr.T @ touch_context, obs_vec_tsr).solution
        # print('c2m_mat_hat:', c2m_mat_hat)
        # input()
        
        # Predict prior of touched VSF
        K_prior = self.prior_factory.predict(vsf)
        
        K_mean = torch.abs(K_prior.param_mean())
        K_var = K_prior.param_std()**2

        hat_vec_tsr = obs_mat_tsr @ K_mean.reshape(-1, 1)
        assert hat_vec_tsr.shape == obs_vec_tsr.shape

        res = obs_vec_tsr - hat_vec_tsr
        if verbose:
            print('res:', (0.5*res.T @ res).item())

        # plt.clf()
        # plt.hist(res.detach().cpu().numpy())
        # plt.xlabel('residual')
        # plt.hist(K_dist.mu.detach().cpu().numpy())
        # plt.xlabel('K mu')
        # K_sig = K_dist.sig
        # if not K_dist.diag:
        #     K_sig = torch.diag(K_sig)
        # plt.hist(K_sig.detach().cpu().numpy())
        # plt.xlabel('K sig')
        # plt.pause(0.01)
        # plt.show()

        loss_start_time = time.time()
        if sig_form == 'uniform':
            loss = 0.5*res.T @ res
        else:
            start_time = time.time()
            obs_sig = obs_var * torch.eye(num_obs, **tsr_params)

            K_sig = K_var * torch.eye(len(obs_idx), **tsr_params)

            tot_sig = obs_mat_tsr @ K_sig @ obs_mat_tsr.T + obs_sig
            
            # assert the tot_sig matrix is symmetric
            assert torch.allclose(tot_sig, tot_sig.T, atol=1e-6), f"tot_sig is not symmetric: {tot_sig}"
            
            if verbose:
                print('compute sig time:', time.time() - start_time)
            
            # TODO: understand why eigh produces errors sometimes, maybe requires float64?
            # eig_val, eig_vec = torch.linalg.eigh(tot_sig)
            # assert torch.all(eig_val > 0), f"Covariance matrix min eig val:{eig_val.min()}"                

            start_time = time.time()
            if sig_form == 'obs':
                tot_sig = tot_sig.detach()
                loss = 0.5*res.T @ torch.linalg.inv(tot_sig) @ res
            elif sig_form == 'obs+opt':
                quad_form = res.T @ torch.linalg.inv(tot_sig) @ res
                s, logdet = torch.slogdet(tot_sig)
                assert s > 0
                loss = 0.5*quad_form + 0.5*logdet
            
            if verbose:
                print('eval mat inv time:', time.time()-start_time)

        if verbose:
            print('compute loss time:', time.time() - loss_start_time)
            print('loss:', loss.item())

        return loss

