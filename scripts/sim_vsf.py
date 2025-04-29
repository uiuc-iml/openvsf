import numpy as np
import torch
import argparse
import dacite
from klampt import vis
from klampt import WorldModel
from klampt.math import se3
import time

import sys
sys.path.append('..')
from vsf import vsf_from_file, vsf_from_box
from vsf.sim import klamptWorldWrapper
from vsf.sim.constructors import simulator_from_config,SimulatorConfig,WorldConfig,WorldObjectConfig,QuasistaticVSFSimulator
from vsf.sim.sim_state_cache import SimStateCache
from vsf.sensor.constructors import SensorConfig, CalibrationConfig, calibration_from_config
from vsf.sensor.joint_torque_sensor import JointTorqueSensor
from vsf.dataset.constructors import DatasetConfig, dataset_from_config
from vsf.visualize.klampt_visualization import add_sim_to_vis
from vsf.utils.perf import PerfRecorder
from vsf.utils.config_utils import load_config_recursive
from vsf.utils.data_utils import remap_dict_in_seq

DRAW_NORMALS = False
DRAW_SPRINGS = False

def simulate_dataset(sim : QuasistaticVSFSimulator, dataset, dataset_config : DatasetConfig, calibrators : dict,
                     sim_cache_dir:str=None,
                     visualize=True):
    perfer = PerfRecorder()
    
    if visualize:
        visitems = add_sim_to_vis(sim, draw_normals=DRAW_NORMALS, draw_springs=DRAW_SPRINGS)
        vis.addPlot('sensors')
        vis.show()

    if sim_cache_dir:
        sim_cache = SimStateCache(sim)
    else:
        sim_cache = None
    dt = 0.1
    
    control_keys = dataset_config.control_keys if len(dataset_config.control_keys) != 0 else sim.get_control_keys()
    sensor_keys = dataset_config.sensor_keys if len(dataset_config.sensor_keys) != 0 else sim.get_sensor_keys()        

    perfer.start('all')
    for seq in dataset:
        sim.reset()
        #extract the controls and observations from this sequence
        control_seq, observation_seq = remap_dict_in_seq(seq, control_keys, sensor_keys)

        #calibrate the sensors from the sequences
        ncalibrate = 0
        for k,v in calibrators.items():
            sensor = sim.get_sensor(k)
            if sensor is None: raise ValueError(f"Sensor {k} not found in simulator")
            n = v.calibrate(sensor, sim, control_seq, observation_seq)
            ncalibrate = max(n,ncalibrate)
        if ncalibrate > 0:
            print(f'Calibrated sensors from {ncalibrate} time steps')
        
        if sim_cache is not None:
            sim_cache.sequences.append([])

        sim.reset()
        for step_idx, frame in enumerate(seq):
            t0 = time.time()
            perfer.start('sim')
            sim.step(control_seq[step_idx],dt)

            if step_idx >= ncalibrate:
                predictions = sim.measurements()
                if sim_cache is not None:
                    perfer.start('save_sim_data')
                    sim_cache.sequences[-1].append(sim_cache.sim_state(dt*step_idx, dt, control_seq[step_idx], observation_seq[step_idx]))
                    perfer.stop('save_sim_data')
            else:
                predictions = {}

            if not args.novis:
                vis.lock()
                vis.animationTime(dt*step_idx)
                for item in visitems:
                    try:
                        vis.remove(item)
                    except Exception as e:
                        pass

                visitems = add_sim_to_vis(sim, draw_normals=DRAW_NORMALS,draw_springs=DRAW_SPRINGS)
                for sensor,value in observation_seq[step_idx].items():
                    if len(value) > 10:
                        val = np.linalg.norm(value)
                        vis.logPlot('sensors',sensor+' norm',val)
                    else:
                        for i in range(len(value)):
                            vis.logPlot('sensors',sensor+'[%d]'%i,value[i])
                for sensor,value in predictions.items():
                    if len(value) > 10:
                        val = torch.linalg.norm(value).cpu().item()
                        vis.logPlot('sensors',sensor+' est norm',val)
                    else:
                        value = value.cpu().numpy()
                        for i in range(len(value)):
                            vis.logPlot('sensors',sensor+' est [%d]'%i,value[i])
                vis.unlock()
                t1 = time.time()
                time.sleep(max(dt - (t1-t0),0))
                # for sensor in vsf_sim.sensor_list:
                #     if sensor_states[f'{sensor.name}_vsf_idx'].size != 0:
                #         vis_mgr.update_pcd(0, sensor_states[f'{sensor.name}_curr_pts'], colors=[1.0, 0.0, 0.0])
                #         vis_mgr.update_pcd(1, sensor_states[f'{sensor.name}_rest_pts'], colors=[0.0, 1.0, 0.0])
                # all_pcd = sum(vsf_sim.klampt_world_wrapper.get_all_pcd(format='open3d'), o3d.geometry.PointCloud())
                # vis_mgr.update_pcd(2, np.array(all_pcd.points))
                # vis_mgr.update_all(fig_fn=None)
                # # if step_idx == len(angles_seq) - 1:
                # #     o3d.visualization.draw_geometries(vis_mgr.pcd_lst + [obj_mgr.vis_pcd])
                

            perfer.stop('sim')
    
    perfer.stop('all')
    if sim_cache_dir:
        print("Dumping simulation cache and performance data to",sim_cache_dir)
        perfer.dump(sim_cache_dir)
        sim_cache.save(sim_cache_dir)
    
    if visualize:
        vis.show(False)
        vis.kill()

def simulate_manual(sim : QuasistaticVSFSimulator, manual_advance = False):
    assert sim.klampt_world_wrapper.world.numRobots() == 1,"Can currently only simulate a single robot"
    robot = sim.klampt_world_wrapper.world.robot(0)
    visitems = add_sim_to_vis(sim, draw_normals=DRAW_NORMALS, draw_springs=DRAW_SPRINGS)
    vis.addPlot('sensors')
    for name,vsf_obj in sim.vsf_objects.items():
        vis.add('pose_'+str(name),se3.from_ndarray(vsf_obj.pose))
        vis.edit('pose_'+str(name))

    vis.show()
    dt = 0.05
    t = 3.0

    def advance(cmd):
        """Where the meat of the simulation and visualization lie"""
        nonlocal t,dt,visitems,robot
        vis.lock()
        controls = {robot.name:np.array(robot.configToDrivers(cmd))}  #convert command to #drivers and cast to numpy array
        sim.step(controls, dt)
        predictions = sim.measurements()
        vis.unlock()

        #update visualization
        vis.lock()
        vis.animationTime(t)
        for item in visitems:
            try:
                vis.remove(item)
            except Exception as e:
                pass
        #this does most of the heavy lifting
        visitems = add_sim_to_vis(sim, draw_normals=DRAW_NORMALS, draw_springs=DRAW_SPRINGS)
        if len(predictions) > 0:
            for sensor,value in predictions.items():
                if len(value) > 10:
                    #only add norm of prediction
                    val = torch.linalg.norm(value).cpu().item()
                    vis.logPlot('sensors',sensor+' norm',val)
                else:
                    for i,v in enumerate(value.cpu().numpy()):
                        vis.logPlot('sensors',sensor+'[%d]'%i,v)
        vis.unlock()
        t += dt

    #set up a simple looping saw-wave trajectory
    def traj(t):
        cycle = (t % 10.0)/10.0
        amp = 1.0
        cmd = robot.getConfig()
        cmd[2] = (1 - abs(cycle*2 - 1))*amp
        return cmd

    def advance_manual():
        nonlocal t
        vis.lock()
        cmd = traj(t)
        vis.unlock()
        advance(cmd)

    if manual_advance:
        vis.addAction(advance_manual,'Advance',' ')
    else:
        vis.add('target',robot.getConfig(),color=(1,1,0,0.5))
        vis.edit('target')

    while vis.shown():
        for name,vsf_obj in sim.vsf_objects.items():
            pose = vis.getItemConfig('pose_'+str(name))
            vsf_obj.pose = se3.ndarray((pose[:9],pose[9:]))
        if manual_advance:
            time.sleep(dt)
        else:
            cmd = vis.getItemConfig('target')
            t0 = time.time()
            advance(cmd)
            t1 = time.time()
            time.sleep(max(dt -(t1-t0),0))
    vis.kill()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
            
    parser = argparse.ArgumentParser(description='Simulates the interactions between a robot and one or more VSFs. Can simulate from a dataset as well.')
    parser.add_argument('items', nargs='*', help='robots or VSFs to add to the world')
    parser.add_argument('--config', type=str, help='path to a config file defining the world and simulator')
    parser.add_argument('--data_dir', type=str, help='directory containing a dataset')
    parser.add_argument('--sim_cache_dir', type=str, default='', help="a directory to save the dataset's cache for faster batch estimation")
    parser.add_argument('--novis', action='store_true', help='disable visualization')
    parser.add_argument('--manual', action='store_true', help='use manual stepping')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    sim = None
    dataset = None
    if args.config is not None:
        config = load_config_recursive(args.config)
        
        if 'world' not in config:
            raise ValueError("Config file must contain a 'world' section")
        if 'simulator' not in config:
            raise ValueError("Config file must contain a 'simulator' section")
        if 'sensors' not in config:
            raise ValueError("Config file must contain a 'sensors' section")
        if 'dataset' not in config:
            raise ValueError("Config file must contain a 'dataset' section")
        world_config = WorldConfig({k:dacite.from_dict(WorldObjectConfig,v,config=dacite.Config(strict=True)) for (k,v) in config['world'].items()})
        sim_config = dacite.from_dict(SimulatorConfig, config['simulator'],config=dacite.Config(strict=True))
        sensors = {}
        for (k,v) in config['sensors'].items():
            if 'name' not in v:
                v['name'] = k
            else:
                assert v['name'] == k, "Sensor name does not match key"
            sensors[k] = dacite.from_dict(SensorConfig,v,config=dacite.Config(strict=True))
        calibrators = {k:calibration_from_config(dacite.from_dict(CalibrationConfig,v,config=dacite.Config(strict=True))) for (k,v) in config.get('calibrators',{}).items()}
        sim = simulator_from_config(sim_config, world_config, sensors)

        if args.data_dir:
            assert len(sim.vsf_objects) > 0,"Simulator does not have any objects associated with it"
            dataset_config = dacite.from_dict(DatasetConfig,config['dataset'],config=dacite.Config(strict=True))
            dataset_config.path = args.data_dir
            dataset = dataset_from_config(dataset_config)

            if config.get('train_sequences',[]):
                dataset = [dataset[i] for i in config['train_sequences']]
    else:
        if args.data_dir:
            print("Cannot load a dataset without a config file")
            parser.print_help()
            sys.exit(1)
        ROBOT_FILE = '../knowledge/robot_model/kinova_gen3_repaired.urdf'
        vsf_items = []
        for item in args.items:
            if item.endswith('.rob') or item.endswith('.urdf'):
                ROBOT_FILE = item
            else:
                vsf_items.append(item)
        world = WorldModel()
        res = world.loadFile(ROBOT_FILE)
        assert res
        robot = world.robot(0)

        world_wrapper = klamptWorldWrapper.from_world(world)
        #need to sample point clouds for the geometries in the world and convert the geometries to SDFs
        world_wrapper.setup_local_pcd_lst(sample_backend='open3d', dispersion=0.01, num_of_pts=1000)
        world_wrapper.setup_local_sdf_lst(0.01)
        #uncomment this to debug the SDF geometries
        # for i,name in enumerate(world_wrapper.name_lst):
        #     if not world_wrapper.bodies_dict[name].geometry().empty():
        #         world_wrapper.bodies_dict[name].geometry().set(world_wrapper.local_sdf_lst[i])
        sensor_list = []
        #comment this out if you don't want to see the joint torque plot
        sensor_list.append(JointTorqueSensor('joint_torques',robot.name,[robot.link(i).getName() for i in range(1,robot.numLinks())]))
        sim = QuasistaticVSFSimulator(world_wrapper,sensor_list)

        #load VSFs from file or create a uniform box object
        if len(vsf_items) > 0:
            for i,vsf_file in enumerate(vsf_items):
                vsf = vsf_from_file(vsf_file)
                vsf_obj = sim.add_deformable('vsf_'+str(i),vsf)
        else:
            print("No VSFs specified; initializing VSF with filled box")
            vsf = vsf_from_box(np.array([0,0,0]),np.array([0.4,0.4,0.5]),resolution=0.02)
            vsf.stiffness.fill_(1.0)
            vsf_obj = sim.add_deformable('vsf',vsf)
            vsf_obj.pose[:3,3] = [0.3,-0.2,0.1]

    if dataset is None:
        #manual visualization
        simulate_manual(sim, manual_advance=args.manual)
    else:
        simulate_dataset(sim, dataset, dataset_config, calibrators, args.sim_cache_dir, visualize=not args.novis)
