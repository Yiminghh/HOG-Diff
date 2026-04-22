from typing import Dict
import numpy as np


class PersistenceImage:
    """
    A wrapper object for the data housed in the image vectorization of a persistence diagram.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
    bandwidth : double, default = 1.0
        Controls the Gaussian kernel for the probability distribution calculated for each point on the birth/persistence diagram.
        See gudhi documentation for more information.
    weight : function, default = lambda x: 1
        Defines the weight function used to compute the weighted sum of the probability distributions for each point on the bith/persistence diagram, i.e. the persistence surface.
        Default is a constant function. Other common choices are a linear function or a bump function, which put greater emphasis on points with longer persistence.
        See gudhi documentation for more information.
    resolution : List[int, int], default = [20,20]
        The dimensions of the persistence image in pixels.
    _pixels : Dict[int, np.array]
        A dictionary that maps each homology dimension to its persistence image in the form of a np.array.
        Each np.array has shape (resolution[0] * resolution[1], 1), and contains the values of the persistence surface at each pixel.
        Initialized to None, set using setter method.
    """

    def __init__(
        self,
        homology_dims=[0, 1],
        bandwidth=1.0,
        weight=lambda x: 1,
        resolution=[20, 20],
    ):
        """Initializes an object of the PersistenceImage class."""
        self.homology_dims = homology_dims
        self.bandwidth = bandwidth
        self.weight = weight
        self.resolution = resolution

        # Initialize image to None
        self._pixels = None

    @property
    def pixels(self) -> Dict[int, np.array]:
        """Getter for the dictionary of persistence images."""
        return self._pixels

    @pixels.setter
    def pixels(self, pixels: Dict[int, np.array]) -> None:
        """Setter for the dictionary of persistence images."""
        assert type(pixels) == dict
        # Assert length matches resolution
        for dim in self.homology_dims:
            assert len(pixels[dim]) == self.resolution[0] * self.resolution[1]
        self._pixels = pixels

    def get_pixels_for_dim(self, dimension: int) -> np.array:
        """Returns a np.array representing the persistence image for the specified dimension."""
        assert (
            self._pixels != None
        ), "Persistence image pixel values have not yet been added to the PersistenceImage object."
        return self.pixels[dimension]

    def __str__(self):
        """Returns a string representation of the Image object."""
        return f"PersistenceImage object with the following attributes: Homology Dims ({self.homology_dims}), Bandwidth ({self.bandwidth}), Weight ({self.weight}), Resolution ({self.resolution})"
