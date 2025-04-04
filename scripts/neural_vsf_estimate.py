import torch
import argparse
import sys
sys.path.append('.')

from vsf import NeuralVSF
from vsf.sim.constructors import simulator_from_config,SimulatorConfig,WorldConfig
from vsf.dataset.constructors import DatasetConfig, dataset_from_config
from vsf.estimator.neural_vsf_estimator import NeuralVSFEstimator, NeuralVSFEstimatorConfig
from vsf.sensor.constructors import CalibrationConfig, SensorConfig, calibration_from_config, BaseCalibrator
from vsf.visualize.klampt_visualization import vsf_show
from vsf.utils.config_utils import load_config_recursive
import dacite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to the config file')
    parser.add_argument('--data_dir', type=str, help='directory to load dataset')
    parser.add_argument('--out', type=str, help='result file (should be a .pt file)')
    parser.add_argument('--sim_out_dir', type=str, help='directory to save all simulation cache')
    args = parser.parse_args()
    if args.out is None or args.config is None:
        parser.print_help()
        sys.exit(1)

    config = load_config_recursive(args.config)

    world_config = dacite.from_dict(WorldConfig,{'objects':config['world']},config=dacite.Config(strict=True))
    sim_config = dacite.from_dict(SimulatorConfig, config['simulator'],config=dacite.Config(strict=True))
    sensors = {k:dacite.from_dict(SensorConfig,v|{'name':k},config=dacite.Config(strict=True)) for (k,v) in config['sensors'].items()}
    calibrators = {k:calibration_from_config(dacite.from_dict(CalibrationConfig,v,config=dacite.Config(strict=True))) for (k,v) in config.get('calibrators',{}).items()}
    vsf_sim = simulator_from_config(sim_config, world_config, sensors)
    assert len(vsf_sim.vsf_objects) > 0,"Simulator does not have any objects associated with it"
    vsf_model = next(iter(vsf_sim.vsf_objects.values())).vsf   # get the only one vsf object registered in the simulator
    assert isinstance(vsf_model,NeuralVSF)
    
    dataset_config = dacite.from_dict(DatasetConfig,config['dataset'],config=dacite.Config(strict=True))
    dataset_config.path = args.data_dir
    dataset = dataset_from_config(dataset_config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vsf_model.to(device)

    estimator_config = dacite.from_dict(NeuralVSFEstimatorConfig,config['estimator'],config=dacite.Config(strict=True))
    solver = NeuralVSFEstimator(estimator_config)
    
    solver.batch_estimate(vsf_sim, vsf_model, dataset, dataset_config, calibrators=calibrators, verbose=True)

    print("Estimation finished, saving to",args.out)
    vsf_model.save(args.out)

    print("Visualizing the VSF")
    from klampt import vis
    vis.init()
    vsf_show(vsf_model)
    