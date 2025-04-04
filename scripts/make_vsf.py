import argparse
import time
import os
import sys
import argparse
import numpy as np
import open3d as o3d

sys.path.append('..')
from vsf.core.vsf_factory import VSFFactory, VSFFactoryConfig, VSFRGBDCameraFactory, VSFRGBDCameraFactoryConfig
from vsf.constructors import AnyVSFConfig
from klampt import Geometry3D
import dacite
from vsf.utils.config_utils import load_config_recursive

if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser(description='Create empty VSF(s) from a factory configuration and object file(s)')
    parser.add_argument('--factory', type=str, help='VSF factory configuration file')
    parser.add_argument('--file', type=str, help='VSF source PCD or geometry file')
    parser.add_argument('--files', type=str, help='folder / text file / JSON / YAML file containing VSF source PCD or geometry files')
    parser.add_argument('--out', type=str, help='output VSF file or folder')
    parser.add_argument('--verbose', action='store_true', help='verbose object creation')
    parser.add_argument('--debug', action='store_true', help='visually debug object creation')
    args = parser.parse_args()
    if args.out is None or args.factory is None or (args.file is None and args.files is None):
        parser.print_usage()
        exit(1)

    factory_json = load_config_recursive(args.factory)
    try:
        factory_cfg = dacite.from_dict(VSFRGBDCameraFactoryConfig,factory_json)
        if args.verbose:
            factory_cfg.verbose = 1
        if args.debug:
            factory_cfg.debug = 1
        factory = VSFRGBDCameraFactory(factory_cfg)
    except Exception:
        factory_cfg = dacite.from_dict(VSFFactoryConfig,factory_json)
        if args.verbose:
            factory_cfg.verbose = 1
        if args.debug:
            factory_cfg.debug = 1
        factory = VSFFactory(factory_cfg)

    if args.file is not None: #single file
        if isinstance(factory,VSFRGBDCameraFactory):
            vsf = factory.process(args.file)
        else:
            vsf = factory.process(args.file)

        print("Saving VSF to file",args.out)
        vsf.save(args.out)
    else:
        if os.path.isdir(args.files):
            files = [os.path.join(args.files,f) for f in os.listdir(args.files)]
        elif args.files.endswith('.txt'):
            files = [l.strip() for l in open(args.files,'r')]
        elif args.files.endswith('.json') or args.files.endswith('.yaml'):
            files = load_config_recursive(args.files)
        else:
            raise ValueError("Invalid files argument "+args.files)
        if len(files) == 0:
            raise ValueError("No files found")
        
        if not os.path.exists(args.out):
            os.makedirs(args.out)
        print("Processing",len(files),"files")
        for i,f in enumerate(files):
            if isinstance(factory,VSFRGBDCameraFactory):
                pcd = o3d.io.read_point_cloud(f)
                vsf = factory.process(pcd)
            else:
                geom = Geometry3D()
                if not geom.loadFile(f):
                    print("Could not load geometry from file",f)
                    continue
                vsf = factory.process(geom)
            outname = os.path.splitext(os.path.basename(f))[0] + '.vsf'
            outpath = os.path.join(args.out,outname)
            print("Saving VSF to file",outpath)
            vsf.save(outpath)
