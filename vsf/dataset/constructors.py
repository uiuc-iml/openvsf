from typing import Dict, Optional, List
from dataclasses import dataclass,field
from .base_dataset import BaseDataset
from .multi_modal_dataset import MultiModalDataset
from .file_loader_dataset import FileLoaderDataset

@dataclass
class DatasetConfig:
    """Configuration of a dataset, for the two types of dataset we support."""

    type: str = ''
    """can be 'file_loader' or empty"""
    path: str = ''
    """where the dataset lies"""
    keys: dict =field(default_factory=dict)
    """key definition for MultiModalDataset"""
    cache: bool = True
    """Whether to cache data in memory. Only used for MultiModalDataset"""
    sensor_keys : Dict[str,str] = field(default_factory=dict) 
    """the keys in the dataset used for matching sensor measurements"""
    control_keys : Dict[str,str] = field(default_factory=dict)
    """the keys in the dataset used for the controls"""


def dataset_from_config(config : DatasetConfig) -> BaseDataset:
    """Returns a BaseDataset properly configured from the given config."""
    if config.type == 'file_loader':
        return FileLoaderDataset(config.path, list(config.keys.keys()))
    else:
        dataset = MultiModalDataset(config.keys,config.path,config.cache, 
                                    config.sensor_keys, config.control_keys)
        return dataset
    