import sys
sys.path.append('..')
from vsf.constructors import vsf_from_file
from vsf.visualize.klampt_visualization import vsf_show
import torch
from klampt import vis

if len(sys.argv) < 1:
    print('Usage: python vis_vsf.py FILE [STIFFNESS_LEVELS] [FEATURE]')
    print("STIFFNESS_LEVELS is a comma-separated list of stiffness levels to visualize")
    print("FEATURE is a feature to visualize.  If it is of the form 'feature:idx'.")
    print("   The visualizer will show the idx'th entry of the feature vector.")
    print("   To show colored points, use commas like 'feature:0,1,2'")
    sys.exit(1)

vsf_model = vsf_from_file(sys.argv[1])

levels = 'auto'
feature = None
feature_idx = 0

if len(sys.argv) >= 3:
    if sys.argv[2] != 'auto':
        levels = list(map(float, sys.argv[2].split(',')))
if len(sys.argv) >= 4:
    feature = sys.argv[3]
    if ':' in feature:
        feature = feature.split(':')
        if ',' in feature[1]:
            feature_idx = list(map(int, feature[1].split(',')))
            feature = feature[0]
        else:
            feature, feature_idx = (feature[0], int(feature[1]))

if feature is None:
    print("Stiffness range",vsf_model.stiffness.min().item(),vsf_model.stiffness.max().item())

vis.init()
#detect whether the feature is an integer index
if feature is not None and isinstance(feature_idx,int):
    if feature not in vsf_model.features:
        print("Feature",feature,"not found")
        print("Valid features include:",",".join(vsf_model.features.keys()))
        sys.exit(1)
    f = vsf_model.features[feature]
    if feature_idx >= len(f):
        print("Feature index out of bounds")
        sys.exit(1)
    values = f[feature_idx]
    if torch.all(values.to(int).to(values.dtype) == values):  #integer
        vsf_show(vsf_model, levels, type='points', feature=feature, feature_idx=feature_idx,cmap='random')
    else:
        vsf_show(vsf_model, levels, feature=feature, feature_idx=feature_idx)
else:
    vsf_show(vsf_model, levels, feature=feature, feature_idx=feature_idx)