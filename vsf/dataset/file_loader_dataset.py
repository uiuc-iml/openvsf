import os
from .base_dataset import BaseDataset
import numpy as np
import glob
import shutil
import os


class FileLoaderSequence:
    def __init__(self, seq_dir: str, state_keys: list[str], frame_prefix='frame', file_ext='npz'):
        self.seq_dir = seq_dir

        self.state_keys = state_keys

        self.frame_prefix = frame_prefix
        self.load_frame_fns(file_ext=file_ext)

    def load_frame_fns(self, file_ext='npz'):
        frame_format = os.path.join(self.seq_dir, self.frame_prefix + '*.' + file_ext)
        self.frame_fns = sorted(glob.glob(frame_format))

    def __len__(self):
        return self.num_frames()

    def num_frames(self):
        return len(self.frame_fns)
    
    def __getitem__(self, idx) -> dict[str, np.ndarray]:
        return self.get_frame(idx)

    def get_frame(self, idx) -> dict[str, np.ndarray]:
        """
        Reads a .npz file and returns a dictionary of NumPy arrays.

        Parameters:
            idx (int): Index to access the filename from self.frame_fns.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys as state_keys and values as NumPy arrays.
        """
        frame_fn = self.frame_fns[idx]  # Get the filename based on the index
        frame = {}  # Initialize the dictionary

        with np.load(frame_fn) as npz_file:  # Open the .npz file
            for key in self.state_keys:  # Iterate over state_keys
                if key in npz_file:  # Check if the key exists in the file
                    frame[key] = npz_file[key]  # Read the dataset into a NumPy array
                else:
                    raise KeyError(f"Key '{key}' not found in the .npz file: {frame_fn}")

        return frame

    def add_frame(self, state_dict: dict[str, np.ndarray], compress:bool=False):
        """
        Adds a frame to the dataset and saves it in .npz format.

        Parameters:
            state_dict (dict[str, np.ndarray]): A dictionary with keys as state_keys and values as NumPy arrays.
        """
        frame_fn = self.frame_prefix + '_{:0>5}.npz'.format(len(self.frame_fns))  # Generate filename for the new frame
        frame_fn = os.path.join(self.seq_dir, frame_fn)  # Add the directory to the filename
        
        # Save the state dictionary to a .npz file
        # NOTE: save compressed takes less space but more time
        if compress:
            np.savez_compressed(frame_fn, **state_dict)
        else:
            np.savez(frame_fn, **state_dict)

        self.frame_fns.append(frame_fn)  # Update the list of frame filenames

    def get_frame_h5(self, idx) -> dict[str, np.ndarray]:
        """
        Reads an H5 file and returns a dictionary of NumPy arrays.

        Parameters:
            idx (int): Index to access the filename from self.frame_fns.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys as state_keys and values as NumPy arrays.
        """
        frame_fn = self.frame_fns[idx]  # Get the filename based on the index
        frame = {}  # Initialize the dictionary

        import h5py
        with h5py.File(frame_fn, 'r') as h5_file:  # Open the H5 file
            for key in self.state_keys:  # Iterate over state_keys
                if key in h5_file:  # Check if the key exists in the file
                    frame[key] = h5_file[key][...]  # Read the dataset into a NumPy array
                else:
                    raise KeyError(f"Key '{key}' not found in the H5 file: {frame_fn}")

        return frame
        
    
    def add_frame_h5(self, state_dict: dict[str, np.ndarray]):
        """
        Adds a frame to the dataset and saves it in .h5 format.

        Parameters:
            state_dict (dict[str, np.ndarray]): A dictionary with keys as state_keys and values as NumPy arrays.
        """
        frame_fn = self.frame_prefix + '_{:0>5}.h5'.format(len(self.frame_fns))  # Generate filename for the new frame
        frame_fn = os.path.join(self.seq_dir, frame_fn)  # Add the directory to the filename
        
        import h5py
        with h5py.File(frame_fn, 'w') as h5_file:  # Open the H5 file for writing
            for key, value in state_dict.items():  # Iterate through the state dictionary
                h5_file.create_dataset(key, data=value)  # Save each key-value pair as a dataset

        self.frame_fns.append(frame_fn)  # Update the list of frame filenames


class FileLoaderDataset(BaseDataset):
    """
    Dataset class that saves and loads intermediate states from disk.
    """
    
    def __init__(self, data_dir: str, state_keys: list[str], sequence_prefix: str='seq', 
                 frame_prefix: str='frame'):
        """
        Args:
            data_dir (str): Directory to save / load the dataset.
            state_keys (list[str]): List of keys to save in the state dictionary.
            sequence_prefix (str): Prefix for the sequence directories.
            frame_prefix (str): Prefix for the frame filenames.    
        """
        super().__init__()

        self.data_dir:str = data_dir
        self.state_keys:list[str] = state_keys
        self.sequence_prefix:str = sequence_prefix
        self.frame_prefix:str = frame_prefix

        seq_format = os.path.join(self.data_dir, sequence_prefix + '*')
        self.sequence_dirs: list[str] = sorted(glob.glob(seq_format))
                
        self.sequences = [FileLoaderSequence(seq_dir, state_keys, frame_prefix) for seq_dir in self.sequence_dirs]

    def __len__(self) -> int:
        """
        Return the number of state sequences in the dataset.
        """
        return len(self.sequence_dirs)

    def get_sequence(self, seq_idx) -> FileLoaderSequence:
        """
        Return a tuple of control trajectory and observation trajectory
        """
        return self.sequences[seq_idx]

    def new_sequence(self, seq_idx, overwrite=False) -> FileLoaderSequence:
        """
        Create a new sequence and return it.
        
        Inputs: 
            seq_idx (int): Index of the new sequence.
            overwrite (bool): If True, overwrite the existing sequence directory.
        Outputs:
            FileLoaderSequence: New sequence object.
        """
        seq_dir = self.sequence_prefix + '_{:0>5}'.format(seq_idx)
        seq_dir = os.path.join(self.data_dir, seq_dir)
        if seq_dir in self.sequence_dirs:
            if overwrite:
                shutil.rmtree(seq_dir)
            else:
                raise ValueError(f"Sequence directory '{seq_dir}' already exists, \
                                 set overwrite=True to overwrite.")
        else:
            self.sequence_dirs.append(seq_dir)

        os.makedirs(seq_dir, exist_ok=True)
        self.sequence_dirs.append(seq_dir)
        
        sequence = FileLoaderSequence(seq_dir, self.state_keys, self.frame_prefix)
        self.sequences.append(sequence)
        return sequence
        