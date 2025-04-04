import numpy as np
import matplotlib
from matplotlib import cm
from typing import Union, Tuple

from dataclasses import dataclass, field
from typing import List

@dataclass
class O3DViewParams:
    zoom: float = 0.8
    front: List[float] = field(default_factory=lambda :[-0.5, 0.5, -0.5])
    lookat: List[float] = field(default_factory=lambda :[0, 0, 0])
    up: List[float] = field(default_factory=lambda :[0, 0, 1])

    def set(self, o3d_vis):
        ctr = o3d_vis.get_view_control()
        ctr.set_zoom(self.zoom)
        ctr.set_front(self.front)
        ctr.set_lookat(self.lookat)
        ctr.set_up(self.up)


def stiffness_to_color(stiffness : np.ndarray, stiffness_upper=None, stiffness_mask_threshold=1e-2, cmap='YlOrRd') -> np.ndarray:
    """Determines a color map for stiffness values.  Low stiffness values
    under stiffness_mask_threshold are masked out.

    The max value of the colormap is determined by stiffness_upper.  If
    stiffness_upper is None, the 98th percentile of stiffness values above
    stiffness_mask_threshold is used.
    """
    if stiffness_upper is None:
        stiffness_mask = stiffness>stiffness_mask_threshold
        if np.sum(stiffness_mask) == 0:
            return np.zeros((stiffness.shape[0], 3))
        stiffness_upper = np.percentile(stiffness[stiffness_mask], 98)
    if stiffness_upper == 0:
        return np.zeros((stiffness.shape[0], 3))
    cm_norm = cm.colors.Normalize(vmin=0, vmax=stiffness_upper)
    colormap = matplotlib.colormaps[cmap]
    colors = colormap(cm_norm(stiffness))[:, :3]
    return colors

def colorize(values : np.ndarray, v_min=None, v_max=None, cmap : str = 'YlOrRd') -> np.ndarray:
    """Colorizes a 1D array of values according to the given color map.
    
    cmap can also be 'random' in which case unique values are given unique colors.
    """
    if cmap == 'random':
        unique_values = np.unique(values)
        colors = np.random.rand(len(unique_values),3)
        color_map = dict(zip(unique_values,colors))
        return np.array([color_map[v] for v in values])

    # Auto detect min and max values if not provided
    if v_min is None:
        v_min = np.min(values)
    if v_max is None:
        v_max = np.max(values)
    
    cm_norm = cm.colors.Normalize(vmin=v_min, vmax=v_max)
    colormap = matplotlib.colormaps[cmap]
    colors = colormap(cm_norm(values))[:, :3]
    return colors

def autodetect_stiffness_values(values : Union[np.ndarray,Tuple[np.ndarray]], lb=1e-2, percentiles = (5,95), scales=None, Nvals=5) -> np.ndarray:
    """Autodetects stiffness values for one or more stiffness arrays"""
    if not isinstance(values,tuple):
        values = (values,)
    if scales is None:
        scales = [1.0]*len(values)
    if np.sum(np.sum(v>lb*s) for v,s in zip(values,scales)) == 0:
        return np.linspace(0,0,Nvals)
    lowers = [np.percentile(v[v>lb*s],percentiles[0])/s for v,s in zip(values,scales)]
    uppers = [np.percentile(v[v>lb*s],percentiles[1])/s for v,s in zip(values,scales)]
    stiffness_lower = min(lowers)
    stiffness_upper = max(uppers)
    return np.linspace(stiffness_lower,stiffness_upper,Nvals)


    