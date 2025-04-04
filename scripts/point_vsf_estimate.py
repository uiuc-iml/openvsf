import numpy as np
import dacite
import argparse
import tqdm
import pickle
import json

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vsf import PointVSF
from vsf.sim.constructors import simulator_from_config,SimulatorConfig,WorldConfig,WorldObjectConfig,QuasistaticVSFSimulator
from vsf.sim import PointVSFQuasistaticSimBody
from vsf.sim.sim_state_cache import SimStateCache
from vsf.sensor.constructors import SensorConfig, CalibrationConfig, calibration_from_config, BaseCalibrator
from vsf.dataset.constructors import DatasetConfig, dataset_from_config, BaseDataset
from vsf.estimator.point_vsf_estimator import PointVSFEstimator, PointVSFEstimatorConfig
from vsf.prior.constructors import make_prior_factory, PointVSFPriorConfig
from vsf.sim.metrics import predict_sensors, rmse_sensors, rmse_from_stats
from vsf.utils.perf import PerfRecorder
from vsf.utils.plot_utils import plot_eval_stats
from vsf.utils.data_utils import to_json_dict
from iml_utils.config import load_config_recursive, save_config
from typing import Tuple

def point_vsf_estimate_batch(vsf_sim : QuasistaticVSFSimulator,
                             dataset : BaseDataset,
                             calibrators : dict[str,BaseCalibrator] = None,
                             vsf_name : str = 'vsf',
                             estimator_config: PointVSFEstimatorConfig = None,
                             prior_config: PointVSFPriorConfig = None,
                             store_sim_data : str = None,
                             perfer : PerfRecorder = None) -> Tuple[PointVSF,PointVSFEstimator]:
    """Standard way of using a recursive estimator to estimate the
    VSF parameters from a dataset. """
    if perfer is None:
        perfer = PerfRecorder()
    
    vsf_sim_body: PointVSFQuasistaticSimBody = vsf_sim.vsf_objects[vsf_name]
    assert isinstance(vsf_sim_body,PointVSFQuasistaticSimBody)
    vsf_model = vsf_sim_body.vsf_model
    assert isinstance(vsf_model,PointVSF)
    
    prior_factory = make_prior_factory(prior_config)
    vsf_estimator = PointVSFEstimator(estimator_config, prior=prior_factory)
    
    perfer.start('all')
    vsf_estimator.batch_estimate(vsf_sim, vsf_model, dataset, dataset_config, calibrators, store_sim_data)
    perfer.stop('all')
    
    if store_sim_data:
        print("Dumping performance data to",store_sim_data)
        perfer.dump(store_sim_data)

    return [vsf_model,vsf_estimator]


def point_vsf_estimate_recursive(vsf_sim : QuasistaticVSFSimulator,
                                 dataset : BaseDataset,
                                 calibrators : dict[str,BaseCalibrator] = None,
                                 vsf_name : str = 'vsf',
                                 estimator_config: PointVSFEstimatorConfig = None,
                                 prior_config: PointVSFPriorConfig = None,
                                 store_sim_data : str = None,
                                 perfer : PerfRecorder = None) -> Tuple[PointVSF,PointVSFEstimator]:
    """Standard way of using a recursive estimator to estimate the
    VSF parameters from a dataset. """
    if perfer is None:
        perfer = PerfRecorder()
    
    vsf_sim_body: PointVSFQuasistaticSimBody = vsf_sim.vsf_objects[vsf_name]
    assert isinstance(vsf_sim_body,PointVSFQuasistaticSimBody)
    vsf_model = vsf_sim_body.vsf_model
    assert isinstance(vsf_model,PointVSF)
        
    if store_sim_data:
        sim_dataset = SimStateCache(vsf_sim)
    else:
        sim_dataset = None
    
    # NOTE: only reads the first sensor
    # sensor_name, sensor_key = list(dataset_config.sensor_keys.items())[0]
    # sensor_dim = dataset_config.keys[sensor_key]
    prior_factory = make_prior_factory(prior_config)
    vsf_estimator = PointVSFEstimator(estimator_config, prior=prior_factory)
    
    vsf_estimator.online_init(vsf_sim, vsf_model)
    
    perfer.start('all')
    for seq in dataset:
        #extract the controls and observations from this sequence
        control_seq = []
        observation_seq = []
        for step_idx in range(len(seq)):
            frame_data = seq[step_idx]
            control_seq.append({k:frame_data[v] for k,v in dataset_config.control_keys.items()})
            observation_seq.append({k:frame_data[v] for k,v in dataset_config.sensor_keys.items()})

        #calibrate the sensors from the sequences
        ncalibrate = 0
        for k,v in calibrators.items():
            sensor = vsf_sim.get_sensor(k)
            if sensor is None: raise ValueError(f"Sensor {k} not found in simulator")
            n = v.calibrate(sensor, vsf_sim, control_seq, observation_seq)
            ncalibrate = max(n,ncalibrate)
        if ncalibrate > 0:
            print(f'Calibrated sensors from {ncalibrate} time steps')
        
        # for sensor in vsf_sim.sensors:
        #     print(f"Sensor {sensor.name} calibration: {sensor.get_calibration()}")

        if sim_dataset is not None:
            sim_dataset.sequences.append([])

        # NOTE: calibration process will change the internal simulator state
        #       reset the simulator state before starting the estimation
        vsf_estimator.online_reset(vsf_sim)
    
        dt = 0.1
        t = dt*n
        for step_idx in tqdm.tqdm(range(ncalibrate,len(seq))):
            perfer.start('step')

            vsf_sim.step(control_seq[step_idx], 0.1)

            #perform the recursive estimation update
            vsf_estimator.online_update(vsf_sim, 0.1, observation_seq[step_idx])

            if sim_dataset is not None:
                sim_dataset.sequences[-1].append(sim_dataset.sim_state(t, dt, control_seq[step_idx], observation_seq[step_idx]))

            perfer.stop('step')
            t += dt

        vsf_estimator.online_finalize(vsf_model)
    perfer.stop('all')
    
    if store_sim_data:
        print("Dumping sim cache to",store_sim_data)
        sim_dataset.save(store_sim_data)
        print("Dumping performance data to",store_sim_data)
        perfer.dump(store_sim_data)

    return [vsf_model,vsf_estimator]


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
            
    parser = argparse.ArgumentParser(description='Estimates Point VSF parameters from a config file and dataset')
    parser.add_argument('--config', type=str, help='path to the config file')
    parser.add_argument('--data_dir', type=str, help='directory containing dataset')
    parser.add_argument('--vsf_model_path', type=str, default=None, 
                        help='path to the VSF model, will overwrite VSF path in the config file')
    parser.add_argument('--batch', action='store_true', help='whether to use batch or online estimation')
    parser.add_argument('--test', action='store_true', help='whether to test an existing estimation')
    parser.add_argument('--sim_cache_dir', type=str, default='', help='a directory to save simulation data for faster batch estimation')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--out', type=str, help='directory to save resulting VSF and statistics')
    parser.add_argument('--save_plot_dir', type=str, default=None, help='directory to save plots') 
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    config = load_config_recursive(args.config)

    if args.seed >= 0:
        import open3d as o3d
        import torch
        np.random.seed(args.seed)
        o3d.utility.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if 'world' not in config:
        raise ValueError("Config file must contain a 'world' section")
    if 'simulator' not in config:
        raise ValueError("Config file must contain a 'simulator' section")
    if 'sensors' not in config:
        raise ValueError("Config file must contain a 'sensors' section")
    if 'prior' not in config:
        raise ValueError("Config file must contain a 'prior' section")
    if 'dataset' not in config:
        raise ValueError("Config file must contain a 'dataset' section")
    world_config = WorldConfig({k:dacite.from_dict(WorldObjectConfig,v,config=dacite.Config(strict=True)) for (k,v) in config['world'].items()})
    sim_config = dacite.from_dict(SimulatorConfig, config['simulator'],config=dacite.Config(strict=True))
    if args.vsf_model_path is not None:
        sim_config.deformables['vsf'].model.path = args.vsf_model_path
    sensors = {}
    for (k,v) in config['sensors'].items():
        if 'name' not in v:
            v['name'] = k
        else:
            assert v['name'] == k, "Sensor name does not match key"
        sensors[k] = dacite.from_dict(SensorConfig,v,config=dacite.Config(strict=True))
    calibrators = {k:calibration_from_config(dacite.from_dict(CalibrationConfig,v,config=dacite.Config(strict=True))) for (k,v) in config.get('calibrators',{}).items()}
    estimator_config = dacite.from_dict(PointVSFEstimatorConfig,config.get('estimator',{}),config=dacite.Config(strict=True))
    sim = simulator_from_config(sim_config, world_config, sensors)
    prior_config = dacite.from_dict(PointVSFPriorConfig,config['prior'],config=dacite.Config(strict=True))
    assert len(sim.vsf_objects) > 0,"Simulator does not have any objects associated with it"
    dataset_config = dacite.from_dict(DatasetConfig,config['dataset'],config=dacite.Config(strict=True))
    dataset_config.path = args.data_dir
    dataset = dataset_from_config(dataset_config)
    
    if args.verbose:
        print("World config:", world_config)
        print("Simulator config:", sim_config)
        print("Sensors:", sensors)
        print("Calibrators:", calibrators)
        print("Prior config:", prior_config)
        print("Estimator config:", estimator_config)
        print("Dataset config:", dataset_config)

    if not args.test:
        if config.get('train_sequences',[]):
            dataset = [dataset[i] for i in config['train_sequences']]
        
        if args.batch:
            model,estimator = point_vsf_estimate_batch(sim, dataset, calibrators, 'vsf',
                                                       estimator_config=estimator_config,
                                                       prior_config=prior_config,
                                                       store_sim_data=args.sim_cache_dir,
                                                       perfer=sim.perfer)
        else:
            model,estimator = point_vsf_estimate_recursive(sim, dataset, calibrators, 'vsf',
                                                           estimator_config=estimator_config,
                                                           prior_config=prior_config,
                                                           store_sim_data=args.sim_cache_dir,
                                                           perfer=sim.perfer)
        print("Saving results to",args.out)
        if not os.path.exists(args.out):
            os.makedirs(args.out)
        save_config(os.path.join(args.out,'config.yaml'), config)
        estimator.vsf.save(os.path.join(args.out,'vsf.npz'))
        sim.perfer.dump(args.out)

        observed,predicted = predict_sensors(sim, dataset, dataset_config, calibrators)
        print("Saving predictions to",os.path.join(args.out,'train_multi_seq_stats.pkl'))
        pickle.dump(predicted, open(os.path.join(args.out,'train_multi_seq_stats.pkl'), 'wb'))

        print("RMSEs:", json.dumps(to_json_dict(rmse_from_stats(observed,predicted)), indent=2))
        
        if args.save_plot_dir is not None:
            print('Saving plots to',args.save_plot_dir)
            plot_eval_stats(observed, predicted, args.save_plot_dir, 
                            plot_per_channel=True, verbose=False)
    else:
        vsf_body: PointVSFQuasistaticSimBody = sim.vsf_objects['vsf']
        vsf_body.vsf_model.load(os.path.join(args.out,'vsf.npz'))
        
        if config.get('test_sequences',[]):
            dataset = [dataset[i] for i in config['test_sequences']]

        observed,predicted = predict_sensors(sim, dataset, dataset_config, calibrators)
        print("Saving predictions to",os.path.join(args.out,'test_multi_seq_stats.pkl'))
        pickle.dump(predicted, open(os.path.join(args.out,'test_multi_seq_stats.pkl'), 'wb'))
        sim.perfer.dump(args.out)

        print("RMSEs:", json.dumps(to_json_dict(rmse_from_stats(observed,predicted,)), indent=2))
        
        if args.save_plot_dir is not None:
            print('Saving plots to',args.save_plot_dir)
            plot_eval_stats(observed, predicted, args.save_plot_dir, 
                            plot_per_channel=True, verbose=False)