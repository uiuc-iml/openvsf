<div align="center">
  <img width="500px" src="https://github.com/user-attachments/assets/e7998656-70dd-4b52-8ea1-c17103ef9450"/>
</div>

---
[![PyPI - Test Release](https://img.shields.io/badge/Test%20PyPI-vsf-informational)](https://test.pypi.org/project/openvsf/)
[![Docs - latest](https://readthedocs.org/projects/openvsf/badge/?version=latest)](https://openvsf.readthedocs.io/en/latest/)

```shell
pip install openvsf  # Install OpenVSF library
```

# Volumetric Stiffness Field (VSF) Library

This library provides an implementation of the Volumetric Stiffness Field (VSF), a model for heterogeneous deformable objects that enables real-time estimation and simulation.  

![Gallery](https://github.com/user-attachments/assets/321282dd-1a10-4ecf-a815-58461ca53ef9)

<p align="center">
  <img src="https://github.com/user-attachments/assets/89276089-0bc5-420b-91a5-0b90306167f7" width=45% style="margin:1px;"/>
  <img src="https://github.com/user-attachments/assets/b4f084b3-bc0a-49c5-beb8-0135b3d11392" width=21% style="margin:1px;"/>
  <img src="https://github.com/user-attachments/assets/7006b119-5608-44cc-9e68-87938c4a453c" width=31% style="margin:1px;"/>
</p>

It provides two types of VSF models:
- Point VSF: a dense set of particles attached to their reference positions with a Hookean spring.
- Neural VSF: a continuous field of stiffness that accumulates Hookean resistance to points traveling through it.

Core features include:
- VSF file I/O
- VSF visualization
- VSF estimation from tactile observations: both batch (offline dataset) and online estimation are implemented.
- VSF interaction simulation: produces estimated tactile observations for a moving robot and sensors.
- VSF meta-learning: priors that map feature vectors (e.g., color, visual features) to stiffness can be learned.
- VSF physics simulation: treat VSFs as deformable assets in quasistatic physics simulators.  
- Point <-> neural VSF convertion through "distillation".
- Sensor model unification, point/neural VSF can run on all provided sensors (joint torque, Punyo pressure and Punyo dense force).
- Custom robot and sensor representations

Features planned soon:
- Rigid body simulation
- Efficiency optimization: 
    + speedup neural VSF estimation by scatter dataset to vertex contact forces level
    + use sparse matrix for Punyo dense forces Jacobian
- Have a standard way to set the initial state for simulations / reset to make sense.

## Installation

### Quick Start

You can install the `openvsf` package directly via `pip`:

```bash
pip install openvsf
```

### Local Installation

We provide a pyproject.toml file, so you can install the project locally in editable (development) mode:

```bash
pip install -e .
```

If you'd like to work with the source code without installation, first ensure the following minimal dependencies are installed:

- `numpy`
- `torch`
- `open3d`
- `dacite`
- `klampt>=0.10.0`

You can install all dependencies including optional packages (e.g., `meshio`, `mesh2sdf`, `cvxpy`) using: `pip install -r requirements.txt`


## Documentation

[Python Manual and API Documentation](https://openvsf.readthedocs.io/en/latest/index.html)

### Building Documentations Locally

1. Install Sphinx:  
    ```bash
    pip install sphinx sphinx-autobuild
    ```
1. Navigate to `docs` and build:
    ```bash
    cd docs && make html
    ```
1. View the generated docs in `build/html/index.html`.


## Walkthrough

The best way to get familiarized with this package is the Jupyter notebooks `demos/vsf_playground.ipynb` and `demos/neural_vsf_playground.ipynb`.  These provide tutorials that walk you through the key steps of the pipeline:

- loading VSFs
- visualization tools
- creating empty VSFs from scratch or RGB-D images
- robot / sensor / simulation setup
- material parameter estimation

The `vsf_playground` walkthrough focuses on point-based VSFs, while `neural_vsf_playground` focuses on neural VSFs.

+ ### File formats

A `PointVSF` instance can be loaded from / saved to a Numpy archive (npz) file or a folder containing Numpy files. 
- The Numpy representation is a dictionary containing keys `points` (rest point positions, shape Nx3), `K` (stiffness, shape N), and `features`.  The features are a dictionary from feature names to length N vectors or NxF shaped matrices.  
- The folder representation contains matrices in `points.npy` (shape Nx3), `K.npy` (shape N), and other `[X].npy` (shape N or NxF) files, where `[X]` becomes a feature name.

A `NeuralVSF` instance can be loaded from / saved to a Pytorch file (extension `.pt`).

Either type is naturally loaded with the `model.load(file_or_folder)` function.  You can then call `model.save(file_or_folder)` to save it to disk.

A *dataset* consists of a list of sequences, where each sequence is a list of dictionaries mapping strings to Numpy arrays.  Each sequence can have its own length and contents, but at a minimum should contain a set of *controls* and a set of *sensor observations*.  Missing controls and sensors are not currently supported.

The `BaseDataset` base class declares this structure, but users will typically not use this directly. To help you format easily loaded datasets, the `MultiModalDataset` class provides a "standard" representation consisting of:
- A folder containing sequence folders of the form `seq_[ID]/`.
- Each sequence folder contains a set of `npy` or image files.
- A numpy file `[item].npy` is a matrix of length T x N (variable)
- Image files are named `[item]_[number].png` where `number` goes from 0 to T-1
- Each entry of the sequence must have the same length (leading dimension T).

+ ### Simulation

Simulators consist of:
- Robots, consisting of articulated links (each a rigid body)
- Rigid bodies
- Deformable bodies (VSF models + a pose)
- Sensors

We currently provide only a single simulation method, the `QuasistaticVSFSimulator`.  This simulator assumes that robots / rigid bodies are controlled directly through their configuration / pose, and that VSFs are either fixed in space or moved via controls.  That is, the simulator will not determine how bodies move through physics; a rigid body will not be pushed by the robot or interactions with other bodies, and a VSF's pose will not be simulated through interactions.  However, you can control the movements of bodies manually.  We hope to relax this assumptions in future versions of this package.

A simulation's functionality consists of:
- Populating robots and rigid objects by loading assets or a Klampt WorldModel.
- Adding VSFs
- Adding sensors
- Resetting state
- Stepping the state
- Predicting sensor observations for the current state
- Loading / saving state

Inside the step function, the simulator applies the controls to advance the bodies in the world, performs collision detection, and calculates contact forces and responses.

For speed, we use a simplified stick-slip contact model for point VSFs and a sticking-only contact model for neural VSFs.  Each point is also assumed to move independently from its neighbors.  This strategy leads to "sticky-particle" artifacts for "digging" motions, because points will be attached to interacting geometry as the geometry is pulled away. Such behavior may be plausible for fluids or excavation but most elastic objects don't behave in such a manner.  For best results, touches should move in relatively straight directions and then reverse course.

+ ### Estimation

We provide VSF material estimation functionality given tactile sensor data.  The sensor is moved through space through one or more sequences, either connected to a robot or a rigid body.  The corresponding observations as the sensor is moved are provided.  We assume that the base pose of the VSF object is fixed or estimated from some other source.  The `vsf.estimation.PointVSFEstimator` and `vsf.estimation.NeuralVSFEstimator` classes provide this functionality.

Estimation can be performed in batch (offline) or online mode.  In offline mode, you prepare a dataset and a data dictionary (`DatasetConfig`) to the estimator all at once and receive the optimized estimate.  In online mode, you will:
- initialize the estimator
- do calibration at the start of each sequence, if needed 
- reset the estimator
- update it for each observation
- finalize it when you wish to extract the optimized parameters.

For highest estimation accuracy, it is important to establish a good sensor calibration, accurate observation model and noise matrices, and a good prior.  Priors are covered in the next section.

In a batch estimation, calibration can be performed per-sequence using a `BaseCalibrator` instance, most often a `TareCalibrator`.  For online estimation, you will operate any calibration you need yourself.

For the observation model, your sensor (the `BaseSensor` instance) will need to be designed properly.  We will cover this in a later section.  We current assume constant, homoskedastic observation noise (although the underlying optimization algorithms do support time-varying noise).


+ ### Material Priors

It is helpful for estimation to start with a general sense of the material distribution of the object.  Also, it is helpful if you have any information about how the stiffness in areas on the object are correlated.  These forms of knowledge are given by *priors* and *meta-priors*, which initialize the estimation processes.  

> **Note**: Neural VSFs do not yet support priors.

A *prior factory* is an object that instantiates priors for a given VSF model.  The simplest initialization is to assign a uniform Gaussian distribution, which is what the standard `GaussianVSFPriorFactory` does.  However, if you want to do something more sophisticated, you can meta-learn a function that maps each point's features to its stiffness distribution.  The `LearnableVSFPriorFactory` will let you such a function if you have estimated multiple VSF models. 

To learn priors, you will set up a PyTorch `torch.nn.Module` mapping a BxF batch of features to a length-B batch of scalar means, and another module mapping features to variance.  Then, you will construct the prior with `prior = LearnableVSFPriorFactory(feature_keys,TorchConditionalDistribution(mean_module,var_module))`.  Calling `prior.meta_learn([vsf1,vsf2,...])` will then learn the prior from the previously estimated VSFs.

> **Note**: Make sure that the feature keys are all present in the VSFs' features, and that the shape of the concatenated feature tensor is NxF. 

+ ### Material Meta-Priors

A meta-prior is a more advanced assumption about how stiffnesses are correlated across the object.  As an example, a homogeneity assumption biases estimates so that touches on one part of the object transfer information about the stiffness to another part of the object.  But, they can also be much more sophisticated and meta-learned from estimated VSFs.  On-line estimators will use meta-prior factories to construct meta-priors for each VSF they are given.

As an example, a `HomogeneousVSFMetaPriorFactory` is given a mean and standard deviation of a homogenous prior.  A material estimation can combine this with a standard VSF prior factory to give combination of heterogeneous + homogeneous terms in the prior, as in the following:

```python
homogeneous_mean = 1.0
homogeneous_std = 2.0
heterogeneous_std = 0.1
estimator = PointVSFEstimator(GaussianVSFPriorFactory(0.0,heterogeneous_std**2), HomogeneousVSFMetaPriorFactory(homogeneous_mean,homogeneous_std**2))
``` 

Here the heterogeneous variance is quite small, which means that the output stiffness will be much closer to a uniform distribution than if the variance was large.

TODO: describe the process of meta-learning for meta-priors.


+ ### Utility Scripts

These are found in the `scripts/` folder and perform a standard set of functions.
- `vis_vsf.py`: shows a VSF loaded from disk.
- `make_vsf.py`: uses a configured factory to save one or more VSFs (with uninitialized material parameters). 
- `sim_vsf.py`: simulates interactions between a robot and one or more VSFs.  The robot can be controlled by the mouse or replay a dataset.
- `point_vsf_estimate.py`: estimates material parameters for a point-based VSF from a standard dataset.  Can also predict / save sensor measurements.
- `neural_vsf_estimate.py`: estimates material parameters for a neural VSF from a "standard" dataset.  Can also predict / save sensor measurements.

+ ### Numerical data types

External interfaces generally use Numpy arrays.  Internal computations generally use PyTorch tensors when possible.  Which representation is expected will be annotated with type hints.

To run material estimation on a different GPU, you should run `vsf_model = vsf_model.to(device)` before passing it to the estimator.

Simulations cannot perform collision detection on the GPU, so internal simulation state objects convert data back to Numpy arrays for forward kinematics and the collision detection steps.



## Customization

+ ### Sensor models

A tactile sensor model `BaseSensor` represents a model of how your sensor captures tactile data as a function of simulated contact states.  It is designed to be attached to a robot (e.g., joint torque sensors), a robot link, or a rigid body.  We have four built-in sensor types:
- `JointTorqueSensor`: measures a robot's joint torques.  The sensor model includes gravity compensation.
- `ForceTorqueSensor`: measures the force/torque between two links of the robot.
- `PunyoPressureSensor`: measures overall pressure for the Punyo bubble sensor.
- `PunyoDenseForceSensor`: measures a force field on the Punyo bubble sensor.

Each sensor can be customized to a robot environment.  We suggest either creating your sensor yourself, or using the `vsf.sensor.constructors.SensorConfig` data structure and `vsf.sensor.constructors.sensor_from_config()` function to create a sensor from a configuration file. 
- `type` gives the class of the sensor.
- `name` gives the identifier of the sensor, used in datasets and data dictionaries.
- `attachment_name` configuration specifies which body or robot in the simulator this is attached to.
- `link_names` are used in joint torque sensors, and specifies for which links the joint torques are measured.

We assume that Punyo dense force and force-torque sensors report data in the local frame of the attached link / body. 

A sensor model is expected to predict measurements (`predict()`) and optional measurement noise (`measurement_errors()`).  Current implementations assume that measurement noise is isotropic unit noise; if you wish to customize the noise levels you can subclass the sensor and override the measurement noise method.

Before it is used, a sensor can be optionally *calibrated* by an associated `BaseCalibrator` instance.  Most commonly, we assume that force sensors require a tare operation to be performed before contact is made, due to hysteresis and drift.  


+ ### Point VSF setup 

A point VSF can be set up via points sampled in a bounding box, points sampled within a 3D geometry, or points sampled in the occluded region behind an an RGB-D point cloud.  It consists of the following items:

- **Rest Points (`rest_points`)**: An `N x 3` array of rest points, stored as a `torch.Tensor`.
- **Stiffness (`stiffness`)**: An `N`-element array representing the stiffness at each point, stored as a `torch.Tensor`.
- **Stiffness axis mode (`axis_mode`)**: An string indicating how stiffness is stored.  `isotropic` is currently the only supported value.  Other values are kept as placeholders for future stiffness modes.
- **Features (`features`)**: A dictionary mapping feature keys to an `N x [variable]` tensor providing a feature descriptor at each point.  Optional.

Typical features include height, color, depth, etc.  They can also be updated by estimators, which will provide the `K_std` stiffness standard deviation and `N_obs` the number of times a particle is estimated to have been touched. 

A set of VSFs with a consistent set of features can be initialized through a *VSF factory*. For example, the `VSFRGBDCameraFactory` will create a VSF from an RGB-D point cloud extrapolating points through the occluded volume at a given resolution.  It will also produce features for each point in the volume using dense feature extractors.

Currently implemented "standard" features include:
- "height": height above lowermost point.
- "distance_to_centroid": distance to point cloud centroid.
- "signed_distance": available for construction by geometry.
- "color": RGB color channels, available for VSFRGBDCameraFactory.
- "fpfh": Fast Point Feature Histogram, available for VSFRGBCameraFactory.
- "depth": depth behind RGB-D point cloud, available for VSFRGBCameraFactory.


+ ### Neural VSF setup

Neural VSF models consist of an axis-aligned bounding box defining the bounds of the VSF, and a neural network model that produces the stiffness field (and optional other fields).  A Neural VSF is configured by the following parameters:

- **Number of Layers (`num_layers`)**: Specifies the depth of the network.
- **Hidden Dimensions (`hidden_dim`)**: Specifies the size of each hidden layer.
- **Skip Connections (`skip_connection`)**: Boolean indicating if skip connections are used.
- **Output Features (`output_names`)**: The name of each output feature, defualt ['stiffness']. 
- **Output Dimension (`output_dims`)**: The dimensionality of output features, default [1] for stiffness.
- **Encoder**: Configures the method and parameters of the encoding strategy, including options like frequency or hashgrid encoding methods. (TODO: currently hard-coded)

A neural VSF can also contain a signed distance field (SDF) that dictates the inside/outside test for a point.  Stiffness outside the SDF will be assumed to be 0.  Including an SDF will let the neural VSF better predict the stiffness inside the object, since it does not have to learn the discontinuity between the object and air.


+ ### VSF simulator setup

A simulator is set up by adding the following objects:
- **Robots**:  Loaded from a `.urdf` file.
- **Rigid bodies**: Loaded from a geometry file (STL, OBJ, etc).
- **Deformable VSF bodies**: Added manually.
- **Sensors**: Added manually. 

All objects are assumed to have a unique name.

You may customize the deformable body contact simulation behavior when added to the simulator.  Currently available parameters are the friction coefficient and collision detection thresholds for point VSFs, and the sampling density for neural VSF integration. 

TODO: describe collision geometry initialization in klamptWorldWrapper.

To control robot configurations, rigid body poses, and deformable body poses, you will pass in a controls dictionary into the `step(controls,dt)` function.  The dictionary maps the object name to its configuration or 4x4 homogeneous transform, given as a Numpy array.  Anything that does not appear in this dictionary will have its pose stay the same.  


+ ### Adding new sensors

You can create your own tactile sensors by inheriting from `BaseSensor`.  A sensor model should be capable of simulating measurements based on the current contact state in a simulator (that is, given a `SimState` object).  Estimators will also need Jacobian calculation, which can either be done by implementing all functions with native PyTorch operations, or by implementing a manual Jacobian method.  These two routes are as follows.

1. Implement `predict_torch()`; implement `predict()` to just call `predict_torch()`.  Here you will need to be careful to use only torch-differentiable functions. 
    
2. Implement both `predict()` and `measurement_force_jacobian()`.  Here, you may use non-torch functions in your implementation.  However, this takes quite a bit of care to create a correct implementation of the Jacobian.  The Jacobian is a dictionary mapping simulation body pairs to Jacobians of the form `d measurement / d contact forces`.  Each key should be of the form `(sensor_body,other_body)`, and its corresponding Jacobian should have dimension # measurements x # forces x 3. 
        





## References

```
@inproceedings{yao2023icra,
  author={Yao, Shaoxiong and Hauser, Kris},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Estimating Tactile Models of Heterogeneous Deformable Objects in Real Time}, 
  year={2023},
  volume={},
  number={},
  pages={12583-12589},
  doi={10.1109/ICRA48891.2023.10160731}
}

@inproceedings{yao2024structured,
  title={Structured Bayesian Meta-Learning for Data-Efficient Visual-Tactile Model Estimation},
  author={Shaoxiong Yao and Yifan Zhu and Kris Hauser},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024},
  url={https://openreview.net/forum?id=TzqKmIhcwq}
}
```

TODO add Neural VSF paper

## License

TODO: MAKE THIS CHOICE 

Details of the licensing of the library, specifying how it can be used and distributed.
