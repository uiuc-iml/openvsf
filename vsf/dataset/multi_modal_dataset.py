from .base_dataset import BaseDataset
import glob
import os
import time
from pathlib import Path
import cv2
import json
import numpy as np
from typing import Union
    
class MultiModalDataset(BaseDataset):
    """
    A dataset that can store multiple types of data along with metadata.

    The dataset structure consists of a top-level directory containing multiple 
    subfolders, each named with :code:`seq_` followed by a sequence ID.

    Args:
        data_types (dict): Mapping between dataset entry names and their types.
            - `int`: 1D column vector of default NumPy array type.
            - `(int, dtype)`: 1D column vector of specified type.
            - `((shape), dtype)`: N-dimensional array of given shape and type.
            - `dtype` can be any array type or `"img"` for images, 
            (will be saved and loaded as many png files instead of a numpy array).
        dir_path (str): Path to load/save the data, toplevel folder to dump things into.
            If it does not exist, it is created.
            Every time a new sequence is collected, a subfolder is created using the unix timestamp,
            and populated with data (numpy arrays / images).
            This class can write/read multiple sequences within this "toplevel folder".
        cache_data (bool, optional): Whether to cache data in memory. Defaults to True.
        sensor_keys (list, optional): Keys used for matching sensor measurements.
        control_keys (list, optional): Keys used for the controls.

    Attributes:
        data_types (dict): Processed mapping of dataset entry names and their types.
        dir_path (str): The dataset storage directory.
        seq_names (list): List of sequence subfolders.
        cache_data (bool): Whether data is cached in memory.
        seq_cache (dict): Cached data storage.
        sensor_keys (list): Keys used for sensor matching.
        control_keys (list): Keys used for control matching.
    """
    
    IMAGE8_DTYPE = np.uint8
    IMAGE16_DTYPE = np.uint16
    
    def __init__(self, data_types:dict, dir_path:str, cache_data=True, 
                 sensor_keys=None, control_keys=None):
        """
        Initialize a multi-modal dataset.
        
        Inputs:
        - data_types: a dictionary mapping data names to their types.
        - dir_path: the directory where the dataset is stored.
        - cache_data: whether to cache data in memory.
        - sensor_keys: the keys in the dataset used for matching sensor measurements
        - control_keys: the keys in the dataset used for the controls        
        """
        super().__init__()
        self.data_types = {}
        for k, t in data_types.items():
            if isinstance(t, int):
                data_type = ((t,), float)
            elif isinstance(t, list) or isinstance(t, tuple):
                shape, dtype = t
                if isinstance(shape, int):
                    shape = (shape,)
                elif isinstance(shape, list) or isinstance(shape, tuple):
                    shape = tuple(shape)
                data_type = (shape, dtype)

            self.data_types[k] = data_type
        
        self.dir_path = dir_path
        if not os.path.isdir(self.dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        seq_names = glob.glob(os.path.join(self.dir_path, "seq*"))
        if len(seq_names) == 0:
            print("MultiModalDataset: Warning, no sequences found in dataset folder", self.dir_path)
        self.seq_names = sorted(seq_names)

        self.cache_data = cache_data
        self.seq_cache = {}
        self.sensor_keys = sensor_keys
        self.control_keys = control_keys
    

    def __len__(self):
        return len(self.seq_names)

    def create_sequence(self, seq_id=None, ret_id=False):
        """
        Create a new data sequence folder, using the given sequence id.

        NOTE: this routine will error if the given sequence id already exists.
        """
        if seq_id is None:
            seq_id = time.strftime("%m%d_%H%M%S")
        seq_path = os.path.join(self.dir_path, f"seq_{seq_id}")
        Path(seq_path).mkdir(exist_ok=True)
        self.seq_names.append(seq_path)
        if ret_id:
            return (seq_path, seq_id)
        return seq_path

    def get_sequence(self, sequenceIdx: Union[int,str]) -> 'MultiModalDatasetSequence':
        """Returns a sequence as a MultiModalDatasetSequence"""
        if isinstance(sequenceIdx, int):
            seq_path = self.seq_names[sequenceIdx]
        else:
            seq_path = sequenceIdx
        if self.cache_data:
            if seq_path in self.seq_cache:
                return self.seq_cache[seq_path]
        res = MultiModalDatasetSequence(self, seq_path)
        if self.cache_data:
            self.seq_cache[seq_path] = res
        return res


class MultiModalDatasetSequence:
    """An accessor for a single sequence in a MultiModalDataset.
    
    This can be treated like a list in read mode.

    In write mode, you should use append to add new frames. When done,
    call save() to write the data to disk. 
    """
    def __init__(self, dataset : MultiModalDataset, seq_path : str):
        self.data_types = dataset.data_types
        self.seq_path = seq_path
        self.data_seq = { key:[] for key in self.data_types.keys() }
        if len(os.listdir(seq_path)) > 0:
            self._load()

    def __len__(self):
        return self.seq_len
    
    def append(self, data:dict):
        """
        Add a data entry to the current in memory data sequence.

        Params:
        ------------------------------------------------------------------
        data    dict(str, array_like)   Data items with associated names
        """
        assert(self.data_seq.keys() == data.keys())
        for key, data_spec in self.data_types.items():
            # TODO: dtype check?
            shape, dtype = data_spec

            # Lazy coder...
            if dtype == "img8":
                new_data = np.array(data[key], dtype=MultiModalDataset.IMAGE8_DTYPE)
                assert new_data.shape == shape, f"MultiModalDataset::add_data: invalid data shape for key {key}, expected {shape} but got {new_data.shape}"
            elif dtype == "img16":
                new_data = np.array(data[key], dtype=MultiModalDataset.IMAGE16_DTYPE)
                assert new_data.shape == shape, f"MultiModalDataset::add_data: invalid data shape for key {key}, expected {shape} but got {new_data.shape}"
            else:
                new_data = np.array(data[key], dtype=dtype)
                assert new_data.shape == shape, f"MultiModalDataset::add_data: invalid data shape for key {key}, expected {shape} but got {new_data.shape}"
                new_data = np.expand_dims(new_data, axis=0) # Add extra axis for concatenation

            self.data_seq[key].append(new_data)
        self.seq_len += 1

    def __getitem__(self, idx : int) -> dict[str, np.ndarray]:
        """
        Read a single frame from this dataset.
        """
        if idx < 0: idx += self.seq_len
        if idx < 0 or idx > self.seq_len: raise IndexError("Index out of range")

        retval = {}
        for key, data_spec in self.data_types.items():
            shape, dtype = data_spec
            if dtype == "img8" or dtype == "img16":
                if idx >= len(self.data_seq[key]) or self.data_seq[key][idx] is None:
                    #lazy load images
                    file_name = os.path.join(self.seq_path, f"{key}_{idx}.png")
                    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
                    while idx >= len(self.data_seq[key]):
                        self.data_seq[key].append(None)
                    self.data_seq[key][idx] = img
            retval[key] = self.data_seq[key][idx]
        return retval

    def _load(self, prefix = None):
        print("Loading sequence", self.seq_path)
        self.data_seq = {}
        seq_len = None
        for key, data_spec in self.data_types.items():
            shape, dtype = data_spec
            fn = prefix + key if prefix is not None else key

            if dtype == "img8" or dtype == "img16":
                # Skip loading images in batch load. They are saved individually as data comes in
                # Load only metadata. (just the dimensions, and sequence length)
                file_name = os.path.join(self.seq_path, fn+".meta")
                print(file_name)
                with open(file_name, "r") as meta_file:
                    metadata = json.load(meta_file)
                assert shape == tuple(metadata["shape"]), f"MultiModalDataset: Shape mismatch for entry {key}, expected {shape} but got {tuple(metadata['shape'])}"
                assert dtype == metadata["dtype"], f"MultiModalDataset: Data type mismatch for entry {key}, expected {dtype} but got {metadata['dtype']}"
                _seq_len = metadata["count"]
                self.data_seq[key] = [None] * _seq_len
            else:
                # Compressed sequences. One array in each list.
                file_name = os.path.join(self.seq_path, fn+".npy")
                np_seq = np.load(file_name, allow_pickle=True)
                assert shape == np_seq.shape[1:], f"MultiModalDataset: Shape mismatch for entry {key}, expected {shape} but got {np_seq.shape[1:]}"
                self.data_seq[key] = np_seq
                _seq_len = np_seq.shape[0]

            if seq_len is None:
                seq_len = _seq_len
            else:
                assert seq_len == _seq_len, f"MultiModalDataset: Length mismatch for entry {key}, expected {seq_len} but got {_seq_len}"
        self.seq_len = seq_len

    def save(self, prefix=None):
        """
        Save the current data sequence.

        Compresses data arrays and dumps them (one array per file) into a folder,
        """
        seq_path = self.seq_path

        for key, data_spec in self.data_types.items():
            shape, dtype = data_spec
            fn = prefix + key if prefix is not None else key

            if dtype == "img8" or dtype == "img16":
                # Save metadata and images
                file_name = os.path.join(seq_path, fn+".meta")
                with open(file_name, "w") as meta_file:
                    json.dump({"shape": shape, "count": self.seq_len, "dtype": dtype}, meta_file)
                for idx in range(self.seq_len):
                    if self.data_seq[key][idx] is not None:
                        file_name = os.path.join(self.seq_path, f"{key}_{idx}.png")
                        cv2.imwrite(file_name, self.data_seq[key][idx])
            else:
                compressed = np.concatenate(self.data_seq[key])
                # Compressed sequences. One array left in each list.
                file_name = os.path.join(seq_path, fn+".npy")
                np.save(file_name, compressed)
