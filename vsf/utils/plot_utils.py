"""
This script contains utility functions for plotting curves in VSF simulation/estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from typing import List,Dict

def plot_eval_stats(observed: List[Dict[str,np.ndarray]], predicted: List[Dict[str,np.ndarray]], 
                    fig_out_dir : str, plot_per_channel=False, fig_dpi=300, verbose=False):
    """
    Plot the prediction and observation of the tactile sensors.
    
    By default, this function will plot the root mean squared error of the
    prediction and observation for each tactile sensor. 
    
    If plot_per_channel is True, this function will plot the error of each
    channel of the tactile sensor. Be aware this may generate a 
    large number of figures for high-dimensional sensor like Punyo sensor.
    
    Args:
        observed: The observations from func:`predict_sensors`.
        predicted: The predicted observations from func:`predict_sensors`.
        fig_out_dir: The output directory to save the figures.
        plot_per_channel: If True, plot the error of each channel of the tactile sensor.
        fig_dpi: The DPI of the output figure.
        verbose: If True, print the shape of the observed and predicted values.
    """
    assert len(observed) == len(predicted), 'The observed and predicted args must have the same number of sequences.'
    assert len(observed) > 0

    sensor_name_list = list(observed[0].keys())
    Path(fig_out_dir).mkdir(parents=True, exist_ok=True)

    for seq_idx,(obs,pred) in enumerate(zip(observed,predicted)):
        
        for sensor_name in sensor_name_list:
            plt.clf()
            
            # sensor_obs & sensor_pred shape: (num_time_steps, sensor_dim)            
            # Note sensor_dim can be a tuple for high-dimensional sensors like Punyo sensor.
            sensor_obs = obs[sensor_name]
            sensor_pred = pred[sensor_name]
            
            if verbose:
                print('sensor_obs shape:', sensor_obs.shape)
                print('sensor_pred shape:', sensor_pred.shape)
            
            sensor_err = np.linalg.norm(sensor_obs-sensor_pred, axis=1)
            
            plt.plot(sensor_err, label=f'{sensor_name} error, seq {seq_idx}')
            
            fn = f'{sensor_name}_error_seq_{seq_idx}'
            
            plt.xlabel('Time step')
            plt.ylabel('Error')    
            # set DPI of the figure
            plt.gcf().set_dpi(fig_dpi)        
            plt.legend()
            
            plt.savefig(os.path.join(fig_out_dir, fn+'.png'))

            if plot_per_channel:
                num_steps = sensor_obs.shape[0]
                sensor_obs_flat = sensor_obs.reshape(num_steps, -1)
                sensor_pred_flat = sensor_pred.reshape(num_steps, -1)
                
                assert sensor_obs_flat.shape == sensor_pred_flat.shape
                
                num_dof = sensor_obs_flat.shape[1]
                for dof_idx in range(num_dof):
                    plt.clf()
                    plt.plot(sensor_obs_flat[:, dof_idx], label=f'{sensor_name} dof {dof_idx} obs, seq {seq_idx}')
                    plt.plot(sensor_pred_flat[:, dof_idx], label=f'{sensor_name} dof {dof_idx} pred, seq {seq_idx}')
                    plt.xlabel('Time step')
                    plt.ylabel('Value')
                    plt.gcf().set_dpi(fig_dpi)
                    plt.legend()
                    
                    fn = f'{sensor_name}_dof_{dof_idx}_seq_{seq_idx}'
                    plt.savefig(os.path.join(fig_out_dir, fn+'.png'))                