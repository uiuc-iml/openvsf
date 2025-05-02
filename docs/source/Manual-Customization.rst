Customization
=============

Sensor models
-------------

A tactile sensor model ``BaseSensor`` represents a model of how your sensor captures tactile data as a function of simulated contact states. It is designed to be attached to a robot (e.g., joint torque sensors), a robot link, or a rigid body. We have four built-in sensor types:

- ``JointTorqueSensor``: measures a robot's joint torques. The sensor model includes gravity compensation.
- ``ForceTorqueSensor``: measures the force/torque between two links of the robot.
- ``PunyoPressureSensor``: measures overall pressure for the Punyo bubble sensor.
- ``PunyoDenseForceSensor``: measures a force field on the Punyo bubble sensor.

Each sensor can be customized to a robot environment. We suggest either creating your sensor yourself, or using the ``vsf.sensor.constructors.SensorConfig`` data structure and ``vsf.sensor.constructors.sensor_from_config()`` function to create a sensor from a configuration file.

- ``type`` gives the class of the sensor.
- ``name`` gives the identifier of the sensor, used in datasets and data dictionaries.
- ``attachment_name`` configuration specifies which body or robot in the simulator this is attached to.
- ``link_names`` are used in joint torque sensors, and specifies for which links the joint torques are measured.

We assume that Punyo dense force and force-torque sensors report data in the local frame of the attached link / body.

A sensor model is expected to predict measurements (``predict()``) and optional measurement noise (``measurement_errors()``). Current implementations assume that measurement noise is isotropic unit noise; if you wish to customize the noise levels you can subclass the sensor and override the measurement noise method.

Before it is used, a sensor can be optionally *calibrated* by an associated ``BaseCalibrator`` instance. Most commonly, we assume that force sensors require a tare operation to be performed before contact is made, due to hysteresis and drift.

Point VSF setup
---------------

A point VSF can be set up via points sampled in a bounding box, points sampled within a 3D geometry, or points sampled in the occluded region behind an RGB-D point cloud. It consists of the following items:

- **Rest Points (``rest_points``)**: An ``N x 3`` array of rest points, stored as a ``torch.Tensor``.
- **Stiffness (``stiffness``)**: An ``N``-element array representing the stiffness at each point, stored as a ``torch.Tensor``.
- **Stiffness axis mode (``axis_mode``)**: A string indicating how stiffness is stored. ``isotropic`` is currently the only supported value. Other values are kept as placeholders for future stiffness modes.
- **Features (``features``)**: A dictionary mapping feature keys to an ``N x [variable]`` tensor providing a feature descriptor at each point. Optional.

Typical features include height, color, depth, etc. They can also be updated by estimators, which will provide the ``K_std`` stiffness standard deviation and ``N_obs`` the number of times a particle is estimated to have been touched.

A set of VSFs with a consistent set of features can be initialized through a *VSF factory*. For example, the ``VSFRGBDCameraFactory`` will create a VSF from an RGB-D point cloud extrapolating points through the occluded volume at a given resolution. It will also produce features for each point in the volume using dense feature extractors.

Currently implemented "standard" features include:

- ``height``: height above lowermost point.
- ``distance_to_centroid``: distance to point cloud centroid.
- ``signed_distance``: available for construction by geometry.
- ``color``: RGB color channels, available for VSFRGBDCameraFactory.
- ``fpfh``: Fast Point Feature Histogram, available for VSFRGBDCameraFactory.
- ``depth``: depth behind RGB-D point cloud, available for VSFRGBDCameraFactory.

Neural VSF setup
----------------

Neural VSF models consist of an axis-aligned bounding box defining the bounds of the VSF, and a neural network model that produces the stiffness field (and optional other fields). A Neural VSF is configured by the following parameters:

- **Number of Layers (``num_layers``)**: Specifies the depth of the network.
- **Hidden Dimensions (``hidden_dim``)**: Specifies the size of each hidden layer.
- **Skip Connections (``skip_connection``)**: Boolean indicating if skip connections are used.
- **Output Features (``output_names``)**: The name of each output feature, default ``['stiffness']``.
- **Output Dimension (``output_dims``)**: The dimensionality of output features, default ``[1]`` for stiffness.
- **Encoder**: Configures the method and parameters of the encoding strategy, including options like frequency or hashgrid encoding methods. (TODO: currently hard-coded)

A neural VSF can also contain a signed distance field (SDF) that dictates the inside/outside test for a point. Stiffness outside the SDF will be assumed to be 0. Including an SDF will let the neural VSF better predict the stiffness inside the object, since it does not have to learn the discontinuity between the object and air.

VSF simulator setup
-------------------

A simulator is set up by adding the following objects:

- **Robots**: Loaded from a ``.urdf`` file.
- **Rigid bodies**: Loaded from a geometry file (STL, OBJ, etc).
- **Deformable VSF bodies**: Added manually.
- **Sensors**: Added manually.

All objects are assumed to have a unique name.

You may customize the deformable body contact simulation behavior when added to the simulator. Currently available parameters are the friction coefficient and collision detection thresholds for point VSFs, and the sampling density for neural VSF integration.

TODO: Describe collision geometry initialization in ``klamptWorldWrapper``.

To control robot configurations, rigid body poses, and deformable body poses, you will pass in a controls dictionary into the ``step(controls, dt)`` function. The dictionary maps the object name to its configuration or 4x4 homogeneous transform, given as a Numpy array. Anything that does not appear in this dictionary will have its pose stay the same.

Adding new sensors
------------------

You can create your own tactile sensors by inheriting from ``BaseSensor``. A sensor model should be capable of simulating measurements based on the current contact state in a simulator (that is, given a ``SimState`` object). Estimators will also need Jacobian calculation, which can either be done by implementing all functions with native PyTorch operations, or by implementing a manual Jacobian method. These two routes are as follows:

1. Implement ``predict_torch()``; implement ``predict()`` to just call ``predict_torch()``. Here you will need to be careful to use only torch-differentiable functions.

2. Implement both ``predict()`` and ``measurement_force_jacobian()``. Here, you may use non-torch functions in your implementation. However, this takes quite a bit of care to create a correct implementation of the Jacobian. The Jacobian is a dictionary mapping simulation body pairs to Jacobians of the form ``d measurement / d contact forces``. Each key should be of the form ``(sensor_body, other_body)``, and its corresponding Jacobian should have dimension ``# measurements x # forces x 3``.
