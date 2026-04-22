from typing import Dict
import numpy as np
import gudhi as gd


class PersistenceDiagram:
    """A wrapper object for the data housed in a persistence diagram for the specified homology dimensions.
    Main attribute (persistence_points) is a Dict[int, np.array] that maps a homology dimension key to a np.array that contains (birth,death) tuples for all the persistence pairs.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
            Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
    _persistence_points : None or Dict[int, np.array]
        A dictionary that maps the homology dimension to a np.array of its persistence pairs.
        Each np.array contains tuples of (birth, death) values for each persistence pair.
        Note that the attribute homology_dims must be a subset of the list of keys (hom. dims.) in this dictionary.
        Initialized to None, set using setter method.
    """

    def __init__(self, homology_dims=[0, 1]):
        """Initializes an instance of the PersistenceDiagram class."""
        self.homology_dims = homology_dims

        # Initialize empty persistence diagram
        self._persistence_pts = None

    @property
    def persistence_pts(self) -> Dict[int, np.array]:
        """Get the PersistenceDiagram's dictionary of persistence points.
        Returns
        -------
        Dict[int,np.array]
            A dictionary that maps np.arrays of persistence point tuples (values) to each homology dimenion (key). Will return None if persistence_pts have not yet been set.
        """
        return self._persistence_pts

    @persistence_pts.setter
    def persistence_pts(self, points: Dict[int, np.array]) -> None:
        """Set the PersistenceDiagram's dictionary of persistence points.
        Parameters
        ----------
        points: Dict[int, np.array]
            A dictionary that maps np.arrays of persistence point tuples (values) to each homology dimenion (key)
        """
        assert type(points) == dict
        self._persistence_pts = points

    def get_pts_for_dim(self, dimension: int) -> np.array:
        # Returns a np.array of birth, death pairs for the dimension
        assert (
            self.persistence_pts != None
        ), "Persistence points have not been added to the PersistenceDiagram object"
        return self.persistence_pts[dimension]

    def plot(self):
        """Plots the persistence diagram using Gudhi’s built-in functionality."""
        # Converting to format that gudhi accepts: [(dim, (birth, death)), ...)]
        persistence = []
        for dim in self.homology_dims:
            for point in self.get_pts_for_dim(dim):
                persistence.append((dim, (point[0], point[1])))
        # Using built-in functionality
        gd.plot_persistence_diagram(persistence)

    def __str__(self) -> str:
        name = "This is a PersistenceDiagram object with the following (birth, death) pairs: \n\t"
        for dim in self.homology_dims:
            name += "H" + str(dim) + ":"
            for pair in self.get_pts_for_dim(dim):
                name += f"({pair[0]}, {pair[1]})"
            name += "\n\t"
        return name
