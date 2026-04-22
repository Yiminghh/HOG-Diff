import gudhi as gd
import numpy as np
from typing import List, Dict
from .base import TopologicalDistance
from ..representations import (
    PersistenceDiagram,
    PersistenceLandscape,
)


class LandscapeDistance(TopologicalDistance):
    """
    Takes in persistence diagrams, converts them to persistence landscapes, and computes the distance between them.
    This TopologicalDistance subclass supports comparison of distributions (i.e. input of lists of persistence diagrams).

    Attributes
    ----------
    diagram 1 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
    diagram 2 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
    norm : int, default=2.
        Defines what norm will be used for calculations. Default is 2.
    resolution : int, default=1000
        The resolution/number of samples in each landscape function.
    num_functions : int, default=5.
        The number of landscape functions to be computed.
    landscape_transformer : gudhi package object
        The object that powers the transformation of persistence diagrams to persistence landscapes.
    """

    def __init__(self, diagram1, diagram2, norm=2, resolution=1000, num_functions=5) -> None:
        """
        Creates a LandscapeDistance object for two persistence diagrams (or lists of persistence diagrams).

        Parameters
        ----------
        diagram 1 : PersistenceDiagram or List[PersistenceDiagram]
            The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
        diagram 2 : PersistenceDiagram or List[PersistenceDiagram]
            The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
        norm : int, default=2.
            Defines what norm will be used for calculations. Default is 2.
        resolution : int, default=1000
            The resolution/number of samples in each landscape function.
        num_functions : int, default=5.
            The number of landscape functions to be computed.
        """
        super().__init__(diagram1, diagram2, norm)
        self.resolution = resolution
        self.num_functions = num_functions

        self.landscape_transformer = gd.representations.Landscape(
            resolution=resolution, num_landscapes=num_functions
        )

    def supports_distribution(self) -> bool:
        """Indicates support for distributions of persistence diagrams."""
        return True

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Core Methods: Fit & Transform                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self):
        """Computes and returns persistence landscapes from the persistence diagrams, diagram1 and diagram2 (or the average persistence landscapes, if diagram1 and/or diagram2 are distributions.

        Returns
        -------
        avg1 : PersistenceLandscape
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'functions' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (num_functions * resolution, 1) which concatenates the samples from all landscape functions.

        avg2 : PersistenceLandscape
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'functions' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (num_functions * resolution, 1) which concatenates the samples from all landscape functions.
        """
        landscapes1 = self._convert_to_landscape(self.diagram1)
        landscapes2 = self._convert_to_landscape(self.diagram2)

        avg1 = self._average_landscape(landscapes1)
        avg2 = self._average_landscape(landscapes2)
        return avg1, avg2

    def transform(
        self, landscape1: PersistenceLandscape, landscape2: PersistenceLandscape
    ) -> float:
        """Computes the norm-based distance between two persistence landscapes.
        Cannot be executed before fit() method has been run to generate persistence landscapes.

        Parameters
        ----------
        landscape1 : PersistenceLandscape
            A wrapper object for the first persistence landscape to be compared.
            Attribute called 'functions' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (num_functions * resolution, 1) which concatenates the samples from all landscape functions.
        landscape1 : PersistenceLandscape
            A wrapper object for the second persistence landscape to be compared.
            Attribute called 'functions' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (num_functions * resolution, 1) which concatenates the samples from all landscape functions.

        Returns
        -------
        float :
            The norm-based distance between the two given persistence landscapes.
        """
        common_dims = set(landscape1.homology_dims).intersection(landscape2.homology_dims)
        difference = self._subtract_landscapes(landscape1, landscape2)

        distance = sum(self.compute_norm(difference.functions[dim]) for dim in common_dims)
        return distance

    def fit_transform(self) -> float:
        """Generates persistence landscapes for diagram1, diagram2 and computes the distance between them.
        Returns
        -------
        float :
            The distance between diagram1 and diagram2, computed via landscape vectorization.
        """
        avg1, avg2 = self.fit()
        return self.transform(avg1, avg2)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _convert_to_landscape(
        self, diagrams: List[PersistenceDiagram]
    ) -> List[PersistenceLandscape]:
        """Convert each persistence diagram to a persistence landscape."""
        landscapes = []
        for diagram in diagrams:
            landscape = PersistenceLandscape(
                homology_dims=diagram.homology_dims,
                num_functions=self.num_functions,
                resolution=self.resolution,
            )
            landscape_functions = {}
            for dim, points in diagram.persistence_pts.items():
                transformed_points = self.landscape_transformer.fit_transform([points])[0]
                landscape_functions[dim] = transformed_points
            # set to PersistenceLandscape functions attribute
            landscape.functions = landscape_functions
            # add to list of Landscapes
            landscapes.append(landscape)
        return landscapes

    @staticmethod
    def _average_landscape(
        landscapes: List[PersistenceLandscape],
    ) -> PersistenceLandscape:
        """Compute and return the average persistence landscape across multiple landscapes."""
        avg_landscape = LandscapeDistance._initialize_average_landscape(landscapes)
        LandscapeDistance._accumulate_landscape_functions(landscapes, avg_landscape)
        LandscapeDistance._normalize_landscape(avg_landscape, len(landscapes))
        return LandscapeDistance._create_average_landscape_object(landscapes, avg_landscape)

    @staticmethod
    def _initialize_average_landscape(
        landscapes: List[PersistenceLandscape],
    ) -> Dict[int, np.ndarray]:
        """Initialize an average landscape dictionary with zero arrays."""
        avg_landscape = {}
        for dim in landscapes[0].functions.keys():
            avg_landscape[dim] = np.zeros_like(landscapes[0].functions[dim])
        return avg_landscape

    @staticmethod
    def _accumulate_landscape_functions(
        landscapes: List[PersistenceLandscape],
        avg_landscape: Dict[int, np.ndarray],
    ) -> None:
        """Accumulate landscape functions into the average landscape."""
        for landscape in landscapes:
            for dim, functions in landscape.functions.items():
                avg_landscape[dim] += functions

    @staticmethod
    def _normalize_landscape(
        avg_landscape: Dict[int, np.ndarray],
        num_landscapes: int,
    ) -> None:
        """Normalize the average landscape by the number of landscapes."""
        for dim in avg_landscape:
            avg_landscape[dim] /= num_landscapes

    @staticmethod
    def _create_average_landscape_object(
        landscapes: List[PersistenceLandscape],
        avg_landscape: Dict[int, np.ndarray],
    ) -> PersistenceLandscape:
        """Create and return a PersistenceLandscape object from the average landscape."""
        avg_pl = PersistenceLandscape(
            homology_dims=list(avg_landscape.keys()),
            num_functions=landscapes[0].num_functions,
            resolution=landscapes[0].resolution,
        )
        avg_pl.functions = avg_landscape
        return avg_pl

    @staticmethod
    def _subtract_landscapes(
        landscape1: PersistenceLandscape, landscape2: PersistenceLandscape
    ) -> PersistenceLandscape:
        """Subtract two landscapes for each common dimension, returning a PersistenceLandscape that represents the difference."""
        assert (
            landscape1.resolution == landscape2.resolution
            and landscape1.num_functions == landscape2.num_functions
        ), "Cannot subtract landscapes with different resolutions or number of landscape functions."

        common_dims = set(landscape1.homology_dims).intersection(landscape2.homology_dims)
        # Initialize PersistenceLandscape to return
        diff_pl = PersistenceLandscape(
            common_dims,
            num_functions=landscape1.num_functions,
            resolution=landscape1.resolution,
        )
        diff_functions = {}
        for dim in common_dims:
            diff_functions[dim] = landscape1.get_fns_for_dim(dim) - landscape2.get_fns_for_dim(dim)
        diff_pl.functions = diff_functions
        return diff_pl

    def __str__(self):
        return f"LandscapeDistance object between (1) [{self.diagram1}] and (2) [{self.diagram2}]"
