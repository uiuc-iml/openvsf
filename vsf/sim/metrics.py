from ..sim.quasistatic_sim import QuasistaticVSFSimulator
from ..dataset import BaseDataset,DatasetConfig
from ..sensor.base_calibrator import BaseCalibrator
from ..utils.data_utils import remap_dict_in_seq
from typing import Dict, List, Tuple
import numpy as np

def predict_sensors(sim : QuasistaticVSFSimulator,
                    dataset : BaseDataset,
                    dataset_metadata : DatasetConfig,
                    calibrators : Dict[str,BaseCalibrator]=None,
                    dt = 0.1, verbose=False) -> Tuple[List[Dict[str,np.array]],List[Dict[str,np.array]]]:
    """
    Evaluate the tactile observation predictions on a dataset.

    Returns:
        (observed, predicted): The observations in the dataset and and
        predictions from the simulator.  Each are a list of # sequences dictionaries,
        each a map from the sensor name to the TxN measurement sequence where T is
        the length of the sequence.
    """
    control_keys = dataset_metadata.control_keys if len(dataset_metadata.control_keys) != 0 else sim.get_control_keys()
    sensor_keys = dataset_metadata.sensor_keys if len(dataset_metadata.sensor_keys) != 0 else sim.get_sensor_keys()
    
    observed = [{} for _ in range(len(dataset))]
    predicted = [{} for _ in range(len(dataset))]
    for i,seq in enumerate(dataset):
        if verbose:
            print(f'Processing sequence {i} in dataset {len(dataset)}')

        control_seq, sensor_seq = remap_dict_in_seq(seq, control_keys, sensor_keys)

        sim.reset()
        n = 0
        if calibrators is not None:
            for k,calibrator in calibrators.items():
                sensor = sim.get_sensor(k)
                assert sensor is not None
                nk = calibrator.calibrate(sensor, sim, control_seq, sensor_seq)
                n = max(nk,n)
        obs = {s.name:[] for s in sim.sensors}
        pred = {s.name:[] for s in sim.sensors}
        for frameno in range(n,len(seq)):
            if verbose:
                print(f'Predicting frame {frameno}/{len(seq)} in sequence {i}/{len(dataset)}')
            sim.step(control_seq[frameno],dt)
            for s in sim.sensors:
                obs[s.name].append(sensor_seq[frameno][s.name])
                pred[s.name].append(s.predict(sim.state()))
        for s in sim.sensors:
            observed[i][s.name] = np.array(obs[s.name])
            predicted[i][s.name] = np.array(pred[s.name])
    return observed, predicted


def rmse_from_stats(observed: List[Dict[str,np.ndarray]], predicted:List[Dict[str,np.ndarray]],
                    aggregate_seqs = False, aggregate_channels = False) -> Dict[str,np.ndarray]:
    """
    Computes the RMSE from multi-sequence statistics.
    
    The RMSE is first computed separately for each of the N sequences,
    independently for each of the M sensor channels.
    
    If aggregate_seqs and aggregate_channels are True, the RMSE is computed
    over all sequences and channels and the result is a single scalar.

    If aggregate_seqs is True, the RMSE is computed over all sequences but
    separately for each channel.  The result is a M-length array.

    If aggregate_channels is True, the RMSE is computed over all channels but
    separately for each sequence.  The result is a N-length array.

    If both aggregate_seqs and aggregate_channels are False, the RMSE is a
    N x M array.
    """
    assert len(observed) == len(predicted),'Mismatched number of sequences'
    mses:Dict[str, List[np.ndarray]] = {}
    for obs,pred in zip(observed,predicted):
        assert len(obs) == len(pred),'Mismatched number of sensors'
        for sensor_name in obs:
            if sensor_name not in mses:
                mses[sensor_name] = []
            assert sensor_name in pred
            if obs[sensor_name].ndim > 1:
                num_time_steps = obs[sensor_name].shape[0]
                obs[sensor_name] = obs[sensor_name].reshape(num_time_steps,-1)
            assert obs[sensor_name].shape == pred[sensor_name].shape, \
                f'Mismatched shape for sensor {sensor_name}: obs {obs[sensor_name].shape} vs pred {pred[sensor_name].shape}'
            if len(obs[sensor_name]) > 0:
                sensor_mse = np.mean((obs[sensor_name]-pred[sensor_name])**2,axis=0)
                mses[sensor_name].append(sensor_mse.tolist())
    
    rmse = {}
    for sensor_name in mses.keys():
        if aggregate_seqs and aggregate_channels:
            rmse[sensor_name] = np.sqrt(np.mean(mses[sensor_name]))
        elif aggregate_seqs:
            rmse[sensor_name] = np.sqrt(np.mean(mses[sensor_name],axis=0))
        elif aggregate_channels:
            rmse[sensor_name] = np.sqrt(np.mean(mses[sensor_name],axis=1))
        else:
            rmse[sensor_name] = np.sqrt(np.array(mses[sensor_name]))
    return rmse


def rmse_sensors(sim : QuasistaticVSFSimulator,
                dataset : BaseDataset,
                dataset_metadata : DatasetConfig,
                calibrators : Dict[str,BaseCalibrator]=None,
                dt = 0.1,
                aggregate_seqs=False,
                aggregate_channels=False) -> Dict[str,List[float]]:
    """
    Evaluate the tactile observation prediction accuracy on a dataset.

    Returns:
        A dictionary of sensor names to RMSEs.  These are aggregated as
        specified by aggregate_seqs and aggregate_channels (see
        :func:`rmse_from_stats`).
    """
    observed,predicted = predict_sensors(sim,dataset,dataset_metadata,calibrators,dt)
    return rmse_from_stats(observed,predicted, aggregate_seqs, aggregate_channels)