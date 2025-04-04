import numpy as np
import torch
from torch import optim
import time
from typing import List, Dict, Tuple

from .point_vsf_estimator import PointVSFEstimator,PointVSFEstimatorConfig
from ..core.point_vsf import PointVSF
from ..prior.prior_factory import BaseVSFPriorFactory, LearnableVSFPriorFactory
from ..prior.meta_prior_factory import BaseVSFMetaPriorFactory, LearnableVSFPriorFactory
from ..dataset import BaseDataset, DatasetConfig
from ..sim.quasistatic_sim import QuasistaticVSFSimulator
from ..sim.sim_state_cache import SimStateCache
from ..utils.perf import PerfRecorder, DummyRecorder
from ..sensor.base_sensor import ContactState, SimState


class PointVSFMetaLearning:
    """Prior learning and meta-prior learning for PointVSF models.

    There are two ways to do the learning.  The first is to learn the prior
    from estimated stiffnesses.  This doesn't take into account the structure
    of observations that were used to estimate the stiffnesses. 
    
    The second is to learn directly from a set of datasets. This is more
    accurate, but requires more computational effort and orchestration.
    """

    def __init__(self,
                 sim: QuasistaticVSFSimulator,
                 prior : LearnableVSFPriorFactory,
                 meta_prior : BaseVSFMetaPriorFactory = None):
        self.sim = sim
        self.prior = prior
        self.meta_prior = meta_prior

    def learn(self, vsfs : List[PointVSF]):
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

    def learn_datasets(self,
                       datasets : List[BaseDataset],
                       dataset_configs : List[DatasetConfig],
                       vsfs : List[PointVSF],
                       calibrators : List[ContactState] = None,
                       dt = 0.1):
        if self.meta_prior is None:
            #standard estimation + meta learning
            for dataset,config,vsf in zip(datasets,dataset_configs,vsfs):
                config = PointVSFEstimatorConfig()
                est = PointVSFEstimator(config,self.prior)
                est.batch_estimate(self.sim, vsf, dataset, config, calibrators, dt)
                #vsf.stiffness should be updated
            self.prior.meta_learn(vsfs)
        else:
            #meta prior learning: stiffness ~ K + A * phi
            raise NotImplementedError("TODO: meta learning")

    def prior_loss(self, vsf: PointVSF, dataset:SimStateCache, sig_form='obs+opt', verbose=False) -> torch.Tensor:
        """
        TODO: prior_loss function is used to learn meta-prior distribution.
        Currently, it is not used to estimate the VSF stiffness.
        """
        assert sig_form in ['uniform', 'obs', 'obs+opt']

        estimator = PointVSFEstimator(PointVSFEstimatorConfig(), self.prior, self.meta_prior)
        obs_info = estimator.linearize_dataset(dataset, verbose)
        obs_info['obs_idx'] = torch.tensor(obs_info['obs_idx'], dtype=vsf.stiffness.dtype, device=vsf.stiffness.device)
        obs_info['obs_mat'] = torch.tensor(obs_info['obs_mat'], dtype=vsf.stiffness.dtype, device=vsf.stiffness.device)
        obs_info['obs_vec'] = torch.tensor(obs_info['obs_vec'], dtype=vsf.stiffness.dtype, device=vsf.stiffness.device)

        obs_idx = obs_info['obs_idx']
        obs_mat_tsr = obs_info['obs_mat']
        obs_vec_tsr = obs_info['obs_vec']

        num_obs = obs_vec_tsr.shape[0]

        if verbose:
            print('obs_idx shape:', obs_idx.shape)
            print('obs_mat_tsr shape:', obs_mat_tsr.shape)
            print('obs_vec_tsr shape:', obs_vec_tsr.shape)

        # NOTE: debug linear regression
        # c2m_mat_hat = torch.linalg.lstsq(obs_mat_tsr.T @ touch_context, obs_vec_tsr).solution
        # print('c2m_mat_hat:', c2m_mat_hat)
        # input()
        
        K_dist = estimator.prior_predict(vsf)[obs_idx]

        hat_vec_tsr = obs_mat_tsr.T @ K_dist.mu.reshape(-1, 1)
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
            num_obs = obs_vec_tsr.shape[0]
            raise NotImplementedError("Observation noise is not implemented in estimator, should be in linearize_dataset") 
            obs_sig = estimator.obs_sig * torch.eye(num_obs, dtype=vsf.stiffness.dtype, device=vsf.stiffness.device)

            K_sig = K_dist.tofull().var

            tot_sig = obs_mat_tsr.T @ K_sig @ obs_mat_tsr + obs_sig
            # np.save('out_data/debug_data/tot_sig.npy', tot_sig.detach().cpu().numpy())
            # input()
            # if verbose:
            #     print('compute sig time:', time.time() - start_time)
            
            eig_val, eig_vec = torch.linalg.eigh(tot_sig)
            assert torch.all(eig_val > 0), f"min eig val:{eig_val.min()}"                

            start_time = time.time()
            if sig_form == 'obs':
                tot_sig = tot_sig.detach()
                loss = 0.5*res.T @ torch.linalg.inv(tot_sig) @ res
            elif sig_form == 'obs+opt':
                quad_form = res.T @ torch.linalg.inv(tot_sig) @ res
                s, logdet = torch.slogdet(tot_sig)
                assert s > 0
                loss = 0.5*quad_form + 0.5*logdet
            
            # if verbose:
            #     print('eval mat inv time:', time.time()-start_time)

        # if verbose:
        #     print('compute loss time:', time.time() - loss_start_time)

        print('loss:', loss.item())

        return loss
    

