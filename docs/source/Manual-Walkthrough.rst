Walkthrough
===========

The best way to get familiarized with this package is the Jupyter notebooks ``demos/vsf_playground.ipynb`` and ``demos/neural_vsf_playground.ipynb``. These provide tutorials that walk you through the key steps of the pipeline:

- loading VSFs
- visualization tools
- creating empty VSFs from scratch or RGB-D images
- robot / sensor / simulation setup
- material parameter estimation

The ``vsf_playground`` walkthrough focuses on point-based VSFs, while ``neural_vsf_playground`` focuses on neural VSFs.

File formats
------------

A ``PointVSF`` instance can be loaded from / saved to a Numpy archive (npz) file or a folder containing Numpy files. 

- The Numpy representation is a dictionary containing keys ``points`` (rest point positions, shape Nx3), ``K`` (stiffness, shape N), and ``features``. The features are a dictionary from feature names to length N vectors or NxF shaped matrices.
- The folder representation contains matrices in ``points.npy`` (shape Nx3), ``K.npy`` (shape N), and other ``[X].npy`` (shape N or NxF) files, where ``[X]`` becomes a feature name.

A ``NeuralVSF`` instance can be loaded from / saved to a Pytorch file (extension ``.pt``).

Either type is naturally loaded with the ``model.load(file_or_folder)`` function. You can then call ``model.save(file_or_folder)`` to save it to disk.

A *dataset* consists of a list of sequences, where each sequence is a list of dictionaries mapping strings to Numpy arrays. Each sequence can have its own length and contents, but at a minimum should contain a set of *controls* and a set of *sensor observations*. Missing controls and sensors are not currently supported.

The ``BaseDataset`` base class declares this structure, but users will typically not use this directly. To help you format easily loaded datasets, the ``MultiModalDataset`` class provides a "standard" representation consisting of:

- A folder containing sequence folders of the form ``seq_[ID]/``.
- Each sequence folder contains a set of ``npy`` or image files.
- A numpy file ``[item].npy`` is a matrix of length T x N (variable)
- Image files are named ``[item]_[number].png`` where ``number`` goes from 0 to T-1
- Each entry of the sequence must have the same length (leading dimension T).

Simulation
----------

Simulators consist of:

- Robots, consisting of articulated links (each a rigid body)
- Rigid bodies
- Deformable bodies (VSF models + a pose)
- Sensors

We currently provide only a single simulation method, the ``QuasistaticVSFSimulator``. This simulator assumes that robots / rigid bodies are controlled directly through their configuration / pose, and that VSFs are either fixed in space or moved via controls. That is, the simulator will not determine how bodies move through physics; a rigid body will not be pushed by the robot or interactions with other bodies, and a VSF's pose will not be simulated through interactions. However, you can control the movements of bodies manually. We hope to relax these assumptions in future versions of this package.

A simulation's functionality consists of:

- Populating robots and rigid objects by loading assets or a Klampt WorldModel.
- Adding VSFs
- Adding sensors
- Resetting state
- Stepping the state
- Predicting sensor observations for the current state
- Loading / saving state

Inside the step function, the simulator applies the controls to advance the bodies in the world, performs collision detection, and calculates contact forces and responses.

For speed, we use a simplified stick-slip contact model for point VSFs and a sticking-only contact model for neural VSFs. Each point is also assumed to move independently from its neighbors. This strategy leads to "sticky-particle" artifacts for "digging" motions, because points will be attached to interacting geometry as the geometry is pulled away. Such behavior may be plausible for fluids or excavation but most elastic objects don't behave in such a manner. For best results, touches should move in relatively straight directions and then reverse course.

Estimation
----------

We provide VSF material estimation functionality given tactile sensor data. The sensor is moved through space through one or more sequences, either connected to a robot or a rigid body. The corresponding observations as the sensor is moved are provided. We assume that the base pose of the VSF object is fixed or estimated from some other source. The ``vsf.estimation.PointVSFEstimator`` and ``vsf.estimation.NeuralVSFEstimator`` classes provide this functionality.

Estimation can be performed in batch (offline) or online mode. In offline mode, you prepare a dataset and a data dictionary (``DatasetConfig``) to the estimator all at once and receive the optimized estimate. In online mode, you will:

- initialize the estimator
- do calibration at the start of each sequence, if needed
- reset the estimator
- update it for each observation
- finalize it when you wish to extract the optimized parameters

For highest estimation accuracy, it is important to establish a good sensor calibration, accurate observation model and noise matrices, and a good prior. Priors are covered in the next section.

In a batch estimation, calibration can be performed per-sequence using a ``BaseCalibrator`` instance, most often a ``TareCalibrator``. For online estimation, you will operate any calibration you need yourself.

For the observation model, your sensor (the ``BaseSensor`` instance) will need to be designed properly. We will cover this in a later section. We currently assume constant, homoskedastic observation noise (although the underlying optimization algorithms do support time-varying noise).

Material Priors
---------------

It is helpful for estimation to start with a general sense of the material distribution of the object. Also, it is helpful if you have any information about how the stiffness in areas on the object are correlated. These forms of knowledge are given by *priors* and *meta-priors*, which initialize the estimation processes.  

.. note:: Neural VSFs do not yet support priors.

A *prior factory* is an object that instantiates priors for a given VSF model. The simplest initialization is to assign a uniform Gaussian distribution, which is what the standard ``GaussianVSFPriorFactory`` does. However, if you want to do something more sophisticated, you can meta-learn a function that maps each point's features to its stiffness distribution. The ``LearnableVSFPriorFactory`` will let you such a function if you have estimated multiple VSF models. 

To learn priors, you will set up a PyTorch ``torch.nn.Module`` mapping a BxF batch of features to a length-B batch of scalar means, and another module mapping features to variance. Then, you will construct the prior with:

.. code-block:: python

   prior = LearnableVSFPriorFactory(feature_keys,
                                    TorchConditionalDistribution(mean_module, var_module))

Calling ``prior.meta_learn([vsf1,vsf2,...])`` will then learn the prior from the previously estimated VSFs.

.. note:: Make sure that the feature keys are all present in the VSFs' features, and that the shape of the concatenated feature tensor is NxF. 

Material Meta-Priors
--------------------

A meta-prior is a more advanced assumption about how stiffnesses are correlated across the object. As an example, a homogeneity assumption biases estimates so that touches on one part of the object transfer information about the stiffness to another part of the object. But, they can also be much more sophisticated and meta-learned from estimated VSFs. On-line estimators will use meta-prior factories to construct meta-priors for each VSF they are given.

As an example, a ``HomogeneousVSFMetaPriorFactory`` is given a mean and standard deviation of a homogenous prior. A material estimation can combine this with a standard VSF prior factory to give a combination of heterogeneous + homogeneous terms in the prior, as in the following:

.. code-block:: python

   homogeneous_mean = 1.0
   homogeneous_std = 2.0
   heterogeneous_std = 0.1
   estimator = PointVSFEstimator(
       GaussianVSFPriorFactory(0.0, heterogeneous_std**2),
       HomogeneousVSFMetaPriorFactory(homogeneous_mean, homogeneous_std**2)
   )

Here the heterogeneous variance is quite small, which means that the output stiffness will be much closer to a uniform distribution than if the variance was large.

TODO: describe the process of meta-learning for meta-priors.

Utility Scripts
---------------

These are found in the ``scripts/`` folder and perform a standard set of functions.

- ``vis_vsf.py``: shows a VSF loaded from disk.
- ``make_vsf.py``: uses a configured factory to save one or more VSFs (with uninitialized material parameters).
- ``sim_vsf.py``: simulates interactions between a robot and one or more VSFs. The robot can be controlled by the mouse or replay a dataset.
- ``point_vsf_estimate.py``: estimates material parameters for a point-based VSF from a standard dataset. Can also predict / save sensor measurements.
- ``neural_vsf_estimate.py``: estimates material parameters for a neural VSF from a "standard" dataset. Can also predict / save sensor measurements.

Numerical Data Types
--------------------

External interfaces generally use Numpy arrays. Internal computations generally use PyTorch tensors when possible. Which representation is expected will be annotated with type hints.

To run material estimation on a different GPU, you should run ``vsf_model = vsf_model.to(device)`` before passing it to the estimator.

Simulations cannot perform collision detection on the GPU, so internal simulation state objects convert data back to Numpy arrays for forward kinematics and the collision detection steps.
