import sys
sys.path.append('..')

from vsf.sim import klamptWorldWrapper, QuasistaticVSFSimulator
from vsf.sensor.joint_torque_sensor import JointTorqueSensor
from vsf.visualize.klampt_visualization import vsf_show,vsf_to_point_cloud
from vsf.core.vsf_factory import VSFRGBDCameraFactory, VSFRGBDCameraFactoryConfig, ViewConfig
from vsf.sensor.joint_torque_sensor import JointTorqueSensor
from vsf.dataset.constructors import DatasetConfig, dataset_from_config
from vsf.sensor import TareCalibrator
import time
import numpy as np

SHOW_DATASET = False

view = ViewConfig(origin=[-0.55639158,1.04689234,0.52593784])  # need to set the origin of the camera so that volume points can be created
#we will select points within a bounding box 
config = VSFRGBDCameraFactoryConfig('../demo_data/est_rubber_fig_tall_angle00/', fps_num=20000, voxel_size=0.05,
                                    view = view, bbox = [[-1.03, -0.64, -0.22], [-0.29, 0.23, 0.73]], downsample_visual=True, verbose=True)
factory = VSFRGBDCameraFactory(config)
#process() does all the work
vsf_empty = factory.process('../demo_data/rubber_fig_tall_angle00/bg_pcd.pcd')  #the PCD file is the point cloud of the RGBD scene, including the background 
print('{} rest points for VSF model created from RGBD factory'.format(len(vsf_empty.rest_points)))

#create a world
world = klamptWorldWrapper()
world.add_robot('kinova','../knowledge/robot_model/kinova_gen3_repaired.urdf')
world.setup_local_pcd_lst('open3d')
world.setup_local_sdf_lst()
robot = world.world.robot(0)

#create a simulator with the world and a joint torque sensor
sensor = JointTorqueSensor('kinova_joint_torques','kinova',[robot.link(i).name for i in range(1,8)])
sim = QuasistaticVSFSimulator(world, [sensor])
sim.add_deformable('vsf',vsf_empty)

# show the world and VSF
#vis.debug(world=world.world, empty_pc = vsf_to_point_cloud(vsf_empty), origin = view.origin)

keys = {'torques':7,'angles':7}  #describes the keys present in the dataset
dataset_config = DatasetConfig('../demo_data/kinova_joint_torques_dataset/rubber_fig_tall_angle00_trail1/arm',
                               keys,
                               sensor_keys={'kinova_joint_torques':'torques'},
                               control_keys={'kinova':'angles'})
dataset = dataset_from_config(dataset_config)
print("Dataset has {} sequences".format(len(dataset)))

vsf_empty.stiffness.fill_(0.1)  #set a guessed stiffness of the VSF model
calibrator = TareCalibrator()

if SHOW_DATASET:
    for seqno in range(len(dataset)):
        #extract the sequence of controls and observations.  This is boilerplate
        seq = dataset[seqno]
        control_seq = []
        sensor_seq = []
        for frame in range(len(seq)):
            control_seq.append({k:frame[v] for k,v in dataset_config.control_keys.items()})
            sensor_seq.append({k:frame[v] for k,v in dataset_config.sensor_keys.items()})
            
        #run the calibration
        sim.reset()
        n = calibrator.calibrate(sensor,sim,control_seq,sensor_seq)
        #returns the # of samples used in calibration.  Technically should skip this number of frames for estimation
        print("Sequence",seqno,"calibration:",sensor.get_calibration())

        #now, run the simulator and compare the predicted torques to the actual torques
        dt = 0.1  # a guessed time step.  There's no time-dependent functionality in the quasistatic simulator, so this doesn't matter
        diffs = []
        for frameno in range(n,len(seq)):
            sim.step(control_seq[frameno],dt)
            pred = sim.measurements()['kinova_joint_torques']
            actual = sensor_seq[frameno]['kinova_joint_torques']
            assert len(pred) == len(actual)
            diffs.append(pred-actual)
        diffs = np.array(diffs)
        print("Sequence",seqno,"joint torque RMSEs",np.sqrt(np.mean(diffs**2,axis=0)))

        if seqno >= 5: break

from vsf.estimator.point_vsf_estimator import PointVSFEstimator, PointVSFEstimatorConfig
from vsf.prior.prior_factory import GaussianVSFPriorFactory
estimator = PointVSFEstimator(PointVSFEstimatorConfig(estimator='dense_ekf'), GaussianVSFPriorFactory(0.5,1.0))
t0 = time.time()
estimator.batch_estimate(sim, vsf_empty, [dataset[i] for i in range(1)], dataset_config, {sensor.name:TareCalibrator()})
t1 = time.time()
print("Estimation took time",t1-t0)

from vsf.visualize.klampt_visualization import vsf_show
vsf_show(vsf_empty)

print("Saving to test_vsf.npz")
vsf_empty.save("test_vsfs.npz")