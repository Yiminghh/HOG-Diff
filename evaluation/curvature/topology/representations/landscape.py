from typing import Dict
import numpy as np


class PersistenceLandscape:
    """
    A wrapper object for the data housed in a persistence landscape for the specified homology dimensions.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
    num_functions : int, default=5
        The number of landscape functions to compute for each homology dimension.
        These are usually denoted lambda_1, lambda_2, ..., lambda_{num_functions}.
        Recall that for each successive landscape function, the values of the highest peak function at each x-value is removed from consideration.
    resolution: int, default = 1000
        Determines the number of samples for the piece-wise landscape functions (i.e. how many points are calculated).
    _functions : Dict[int, np.array]
        A dictionary that maps the homology dimension to a np.array of its landscape functions.
        Each np.array contains {num_functions * resolution} floats, which are the {resolution} number of samples for all {num_functions} landscape functions concatenated together.
        Initialized to None, set using setter method.
    """

    def __init__(self, homology_dims=[0, 1], num_functions=5, resolution=1000):
        """Initializes an object of the PersistenceDiagram class."""
        self.homology_dims = homology_dims
        self.num_functions = num_functions
        self.resolution = resolution

        # Initialize landscape functions to None
        self._functions = None

    @property
    def functions(self) -> Dict[int, np.array]:
        """Getter for the complete dictionary of landscape functions."""
        return self._functions

    @functions.setter
    def functions(self, functions: Dict[int, np.array]) -> None:
        """Setter for the complete dictionary of landscape functions."""
        assert type(functions) == dict
        for dim in self.homology_dims:
            assert (
                len(functions[dim]) == self.num_functions * self.resolution
            ), f"The length of the landscape function samples ({len(functions[dim])}) is not the product of the specified resolution ({self.resolution}) and number of landscape functions ({self.num_functions})."
        self._functions = functions

    def get_fns_for_dim(self, dimension: int) -> np.array:
        """Returns a np.array of concatenated landscape functions for the specified dimension."""
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        return self.functions[dimension]

    def get_fn_for_dim(self, dimension: int, fn_num: int) -> np.array:
        """Returns a list of points for landscape function associated with the given dimension and function number."""
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        start_idx = fn_num * self.resolution
        indices = range(start_idx, start_idx + self.resolution)
        return self.functions[dimension][indices]

    def _separate_landscape_functions(self) -> Dict[int, np.array]:
        """Returns an amended function dictionary.
        Keys are still homology dimensions, but np.array values hold {num_functions} lists, each with {resolution} floats, as opposed to simply {num_functions * resolution} floats.
        Thus the landscape functions are stored into separate lists within the np.array, rather than being concatenated.
        """
        function_list_by_dim = {}
        for dim in self.homology_dims:
            function_list = []
            for fn_num in range(0, self.num_functions):
                start_idx = fn_num * self.resolution
                indices = range(start_idx, start_idx + self.resolution)
                fn = self.functions[dim][indices]
                function_list.append(fn)
            function_list_by_dim[dim] = function_list
        return function_list_by_dim

    def __str__(self):
        """Returns a string representation of the PersistenceLandscape object."""
        name = f"This is a PersistenceLandscape object with the following attributes: Homology Dimensions ({self.homology_dims}), Number of Landscape Functions per Dimension ({self.num_functions}), resolution i.e. pts/landscape function ({self.resolution})"
        return name
