#!/usr/bin/env python3

import os
import json
import shutil
import argparse
import yaml

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import vsf
from vsf.core.vsf_factory import VSFFactoryConfig, VSFRGBDCameraFactoryConfig, ViewConfig, CleanPCDConfig


def reformat_assets():
    parser = argparse.ArgumentParser(description="Copy files and generate config YAML for items in a JSON file.")
    parser.add_argument("--json", default="knowledge/vsf_database/small_object_params.json",
                        help="Path to the JSON file with items data.")
    parser.add_argument("--dest", default="/media/motion/G14/vsf_release/shoe_asset_dataset", 
                        help="Path to the destination directory.")
    parser.add_argument("--voxel_size", type=float, help="Voxel size for the VSF.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    json_file = args.json
    dest_dir = args.dest

    # 1. Create destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    # 2. Load the JSON data
    with open(json_file, "r") as f:
        items_data:dict[str, dict] = json.load(f)

    # 3. For each entry in the JSON, process
    for item_name, item_info in items_data.items():
        # item_info example:
        # {
        #   "bbox": [[-1.07, -0.50, -0.27], [-0.16, 0.24, 1.1]],
        #   "vg_shape": [100, 100, 200],
        #   "fps_num": 30000,
        #   "max_iters": 20,
        #   "out_dir": "/media/motion/G14/vsf_meta_learn/obj_asset_dataset/cherry_tree_angle02",
        #   "trans_fn": "knowledge/calibraiton/update_trans_params6.json"
        # }

        out_dir = item_info["out_dir"]
        trans_fn = item_info["trans_fn"]  # path to the extrinsic
        bbox = item_info.get("bbox", None)
        vg_shape = item_info.get("vg_shape", None)
        fps_num = item_info.get("fps_num", None)
        max_iters = item_info.get("max_iters", None)

        if args.voxel_size is not None:
            voxel_size = args.voxel_size
        else:
            vx_size = (bbox[1][0] - bbox[0][0]) / vg_shape[0]
            vy_size = (bbox[1][1] - bbox[0][1]) / vg_shape[1]
            vz_size = (bbox[1][2] - bbox[0][2]) / vg_shape[2]
            voxel_size = max(vx_size, vy_size, vz_size)
            print(f"Voxel size: {voxel_size}")

        # The new directory we will copy into:
        target_item_dir = os.path.join(dest_dir, item_name)
        os.makedirs(target_item_dir, exist_ok=True)

        # 1. Copy files: bg_pcd.pcd, color_img.jpg, depth_img.png from out_dir to target
        for filename in ["bg_pcd.pcd", "color_img.jpg", "depth_img.png"]:
            src_file = os.path.join(out_dir, filename)
            dst_file = os.path.join(target_item_dir, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: {src_file} does not exist and cannot be copied.")

        # 2. Copy `trans_fn` to extrinsic.json in target
        extrinsic_src = trans_fn
        extrinsic_dst = os.path.join(target_item_dir, "extrinsic.json")
        if os.path.exists(extrinsic_src):
            shutil.copy2(extrinsic_src, extrinsic_dst)
        else:
            print(f"Warning: {extrinsic_src} does not exist and cannot be copied.")
            
        extrinsic = json.load(open(extrinsic_dst, 'r'))
        camera_center = np.array(extrinsic['cam2rob'])[:3, 3]

        # 3. Create a new file: rgbd_factory_config.yaml
        config_dict = {
            'bbox': bbox,
            'view': {'origin': camera_center.tolist()}
        }
        if fps_num is not None:
            config_dict['fps_num'] = fps_num
            config_dict['downsample_visual'] = True
        if max_iters is not None:
            assert voxel_size is not None
            config_dict['depth_extrapolation'] = max_iters * voxel_size
            print(f"Depth extrapolation: {config_dict['depth_extrapolation']}")
        config_path = os.path.join(target_item_dir, "rgbd_factory_config.yaml")
        with open(config_path, "w") as yaml_file:
            yaml.dump(config_dict, yaml_file, sort_keys=False)

        print(f"Processed '{item_name}' -> {target_item_dir}")

def reformat_joint_torques_dataset(old_root, new_root):
    """
    This function will iterate through `old_root`, which contains multiple
    subdirectories (each representing a specific object). Inside each object
    directory, there is assumed to be a subdirectory named 'arm' containing
    several sequence folders. Each sequence folder may have the files
    `angles.npy`, `torques.npy`, and `velocities.npy` that need to be copied.
    
    It will create a parallel directory structure in `new_root` such that:
    
    Old structure:
    old_root/
      ├─ object1/
      │   └─ arm/
      │       ├─ seq1/
      │       │   ├─ angles.npy
      │       │   ├─ torques.npy
      │       │   └─ velocities.npy
      │       └─ seq2/
      │           └─ ...
      └─ object2/
          └─ arm/
              └─ seq1/
                  └─ ...
    
    New structure:
    new_root/
      ├─ object1/
      │   ├─ seq1/
      │   │   ├─ angles.npy
      │   │   ├─ torques.npy
      │   │   └─ velocities.npy
      │   └─ seq2/
      └─ object2/
          └─ seq1/
    
    :param old_root: Path to the root directory of the old dataset
    :param new_root: Path to where the new dataset should be generated
    """
    
    # Ensure the new root directory exists
    os.makedirs(new_root, exist_ok=True)
    
    # Iterate over every item in `old_root` (these are your "objects")
    for obj_name in os.listdir(old_root):
        obj_path = os.path.join(old_root, obj_name)
        if not os.path.isdir(obj_path):
            continue  # Skip files or anything that is not a directory
        
        # We expect an 'arm' subdirectory in each object's folder
        arm_path = os.path.join(obj_path, "arm")
        if not os.path.exists(arm_path):
            # If there's no 'arm' folder, skip or handle differently as needed
            continue
        
        # Now iterate through each sequence directory inside `arm`
        for seq_name in os.listdir(arm_path):
            seq_path = os.path.join(arm_path, seq_name)
            
            # Confirm it's actually a directory
            if not os.path.isdir(seq_path):
                continue
            
            # Create the corresponding new directory path
            new_obj_path = os.path.join(new_root, obj_name)
            new_seq_path = os.path.join(new_obj_path, seq_name)
            os.makedirs(new_seq_path, exist_ok=True)
            
            # Copy over the specific files we need
            for fname in ["angles.npy", "torques.npy", "velocities.npy"]:
                old_file = os.path.join(seq_path, fname)
                new_file = os.path.join(new_seq_path, fname)
                if os.path.isfile(old_file):
                    shutil.copy2(old_file, new_file)

            # If you need to copy more files or handle JSON differently,
            # you could extend the logic here.

    print(f"Reformatted dataset is now located in: {new_root}")

if __name__ == "__main__":
    reformat_assets()
    # reformat_joint_torques_dataset(
    #     "/media/motion/G14/vsf_meta_learn/arm_data", 
    #     "/media/motion/G14/vsf_meta_learn/plant_joint_torques_dataset"
    # )