from .base_sensor import BaseSensor
import numpy as np
from typing import List, Dict
from .base_sensor import SimState

class BaseCalibrator:
    """A base class for sensor calibration.  For example, a sensor can be tared
    or calibrated to a specific range using this class.
    """

    def compatible(self, sensor : BaseSensor) -> bool:
        """Returns true if the calibrator is compatible with the sensor."""
        raise NotImplementedError()

    def calibrate(self,
                  sensor : BaseSensor,
                  sim,
                  command_sequence : List[Dict[str,np.ndarray]],
                  observation_sequence : List[Dict[str,np.ndarray]]) -> int:
        """Calibrates the sensor based on the data sequence.
        
        The implementer will use the `sensor.set_calibration` method, with the
        appropriate calibration data structure, to set the calibration.

        Note that a calibrator could theoretically run the simulation on
        the entire sequence.  However, this is not recommended as it would
        be using results from the future to calibrate the sensor.  It is
        recommended that only a small fraction of the sequence be used for
        calibration.

        Args:
            sensor (BaseSensor): The sensor to calibrate.
            sim (QuasistaticVSFSimulator): The simulation object.
            command_sequence (List[Dict[str,np.ndarray]]): The command sequence.
            observation_sequence (List[Dict[str,np.ndarray]]): The observation
                sequence.

        Returns:
            int: The number of time steps used for calibration.  Ideally these
            should be skipped for estimation and prediction.
        """
        raise NotImplementedError()


class TareCalibrator(BaseCalibrator):
    """A simple calibrator that tares the sensor based on the average of some
    initial set of observations.
    
    NOTE: the average is not average observation, but the average of the
    observation residual after subtracting simulation with empty VSF model. 
    
    The calibration has the form {output_key: average_observation}
    """
    def __init__(self, num_samples : int = 10, output_key : str = 'tare'):
        self.num_samples = num_samples
        self.output_key = output_key
    
    def compatible(self, sensor):
        return True

    def calibrate(self, sensor, sim, 
                  command_sequence, observation_sequence) -> int:
        nsamples = min(self.num_samples, len(observation_sequence))
        if nsamples == 0:
            return
        average = np.zeros(observation_sequence[0][sensor.name].shape)
        tare_sim_avg = np.zeros(observation_sequence[0][sensor.name].shape)
        # Dynamically set the tare based on whether the command is moving and whether contacts are present
        for i in range(nsamples):
            sim.step(command_sequence[i], dt=0.1)
            sim_state : SimState = sim.state()
            
            # Contact happened, stop the calibration
            if len(sim_state.contacts) > 0:
                if i == 0:
                    # Make sure at least one sample is used for calibration
                    import warnings
                    warnings.warn('Contact happens at the first sample, please check the command sequence')
                else:
                    nsamples = i
                    break

            # Here measurements are not caused by the contact
            average += observation_sequence[i][sensor.name]
            tare_sim = sim.measurements()[sensor.name].detach().cpu().numpy()
            tare_sim_avg += tare_sim
        average /= nsamples
        tare_sim_avg /= nsamples
        sensor.set_calibration({self.output_key:average, 
                                'tare_sim_avg':tare_sim_avg})
        return nsamples
