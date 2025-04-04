# Base class for VSF model

from __future__ import annotations
import torch


class BaseVSF:
    """
    Base class for VSF model.

    The volumetric stiffness field defines the stiffness of a set of Hookean springs
    in a 3D volume. For each spring with deformation u, VSF will generate contact
    force f = -k*u, where k is the stiffness of the spring.
    """

    def __init__(self):
        pass

    def getStiffness(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get stiffness at 3d points.

        Args:
            points: a tensor of shape (N,3)

        Returns:
            stiffness: a tensor of shape (N,)
        """
        raise NotImplementedError

    def getBBox(self) -> torch.Tensor:
        """
        Return the bounding box of the model.
        """
        raise NotImplementedError

    def save(self, path: str):
        """
        Save the VSF model parameters in a folder/file.
        """
        raise NotImplementedError

    def load(self, path: str):
        """
        Load the VSF model parameters from a folder/file.
        """
        raise NotImplementedError

    def to(self, device) -> BaseVSF:
        """Converts the VSF to a given device or dtype.

        Note: modifies the model in-place.
        """
        raise NotImplementedError
