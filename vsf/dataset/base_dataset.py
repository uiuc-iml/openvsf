from typing import Union,Sequence

class BaseDataset:
    """
    Base class for sequential dataset.  Each entry is a sequence of something,
    typically dicts mapping sequence names to numpy arrays.
    """
    def __init__(self):
        pass
    
    def __len__(self) -> int:
        """
        Return the number of sequences in the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, sequenceIdx : Union[int,slice,Sequence[int]]) -> list:
        """Allows accessing sequences with dataset[index] where
        index is an integer, slice, or sequence."""
        if isinstance(sequenceIdx, slice):
            return SubsetDataset(self,sequenceIdx.indices(len(self)))
        elif hasattr(sequenceIdx, '__iter__'):
            return SubsetDataset(self,sequenceIdx) 
        else:
            if sequenceIdx > len(self):
                raise IndexError("Index out of range")
            return self.get_sequence(sequenceIdx)
        
    def get_sequence(self, sequenceIdx: int) -> list:
        """
        Returns the sequence with index `sequenceIdx`.  The sequence
        is assumed to be able to be treated like a list.
        """
        raise NotImplementedError


class SubsetDataset(BaseDataset):
    """
    A dataset that is a subset of another dataset.
    """
    def __init__(self, dataset: BaseDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def get_sequence(self, sequenceIdx: int) -> list:
        return self.dataset.get_sequence(self.indices[sequenceIdx])