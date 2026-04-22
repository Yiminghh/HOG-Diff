from typing import List, Optional

import gudhi as gd
import gudhi.representations
import gudhi.wasserstein
import networkx as nx
import numpy as np
from .representations import PersistenceDiagram

SMALL_NONZERO_FEATURE = [1e-15, 1e-14]


class GraphHomology:
    """
    Compute persistent homology on graphs by filtering over edge attributes.

    This class uses Gudhi's `SimplexTree` to build a clique complex on a given
    graph and compute persistence diagrams for homology groups, using an edge
    attribute (e.g. curvature) as a filtration function.

    Attributes
    ----------
    homology_dims : List[int]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
        Default is [0, 1].

    filter_attribute : str, optional
        The edge attribute to use as the filtration value. Default is "curvature".
    """

    # Small constants to ensure numerical stability and prevent empty persistence intervals

    def __init__(
        self,
        homology_dims: Optional[List[int]] = None,
        filter_attribute: str = "curvature",
        mask_infinite_features: bool = False,
        extended_persistence: bool = True,
    ):
        """
        Initializes the GraphHomology class with parameters for homology computation.

        Parameters
        ----------
        homology_dims : list of int, optional
            Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
            Default is [0, 1].
        filter_attribute : str, optional
            The edge attribute to use as the filtration value. Default is "curvature".
        """
        self.homology_dims = homology_dims if homology_dims is not None else [0, 1]
        self.filter_attribute = filter_attribute
        self.max_dimension = max(self.homology_dims) + 1
        self.mask_infinite_features = mask_infinite_features
        self.extended_persistence = extended_persistence

    def calculate_persistent_homology(self, G: nx.Graph) -> PersistenceDiagram:
        """
        Calculates persistent homology of the graph's clique complex.

        Parameters
        ----------
        G : networkx.Graph
            Input graph with edges that contain attributes for filtration.

        Returns
        -------
        PersistenceDiagram
            A persistence diagram wrapper for the topological information from a curvature filtration.
            Attribute persistence_pts stores a Dict[int, np.array] that a maps homology dimension key to a np.array of its persistence pairs.

        Raises
        ------
        ValueError
            If the specified `filter_attribute` is not found in an edge's attributes.
        """
        simplex_tree = self._build_simplex_tree(G)
        simplex_tree.make_filtration_non_decreasing()
        simplex_tree.expansion(self.max_dimension)
        if self.extended_persistence:
            simplex_tree.extended_persistence()
        else:
            simplex_tree.persistence(persistence_dim_max=True)

        diagrams = self._format_persistence_diagrams(simplex_tree)
        return diagrams

    def _build_simplex_tree(self, G: nx.Graph) -> gd.SimplexTree:
        """
        Builds a simplex tree from the graph's edges and edge attributes.

        Parameters
        ----------
        G : networkx.Graph
            Input graph where edges contain the `filter_attribute` to use as filtration values.

        Returns
        -------
        gudhi.SimplexTree
            Simplex tree built from the graph, using specified edge attribute as filtration.

        Raises
        ------
        ValueError
            If the specified `filter_attribute` is not found in an edge's attributes.
        """
        st = gd.SimplexTree()

        for u, v, w in G.edges(data=True):
            if self.filter_attribute in w:
                weight = w[self.filter_attribute]
                st.insert([u, v], filtration=weight)
            else:
                raise ValueError(
                    f"Edge ({u}, {v}) is missing required attribute '{self.filter_attribute}'."
                )

        return st

    def _format_persistence_diagrams(self, simplex_tree: gd.SimplexTree) -> PersistenceDiagram:
        """
        Converts a gd.SimplexTree into a PersistenceDiagram object.

        Parameters
        ----------
        simplex_tree: gd.SimplexTree
            Simplex tree built from the graph by conducting a filtration.

        Returns
        -------
        PersistenceDiagram
            A persistence diagram wrapper for the topological information from a curvature filtration.
            Attribute persistence_pts stores a Dict[int, np.array] that a maps homology dimension key to a np.array of its persistence pairs.
        """
        # initialize PersistenceDiagram object
        diagram = PersistenceDiagram(self.homology_dims)
        # format dictionary of mapping persistence points to homology dimensions for input into PersistenceDiagram object
        persistence_pts = {}
        for dim in self.homology_dims:
            if self.mask_infinite_features:
                persistence_pairs = self._mask_infinities(
                    simplex_tree.persistence_intervals_in_dimension(dim)
                )
            else:
                persistence_pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            persistence_pts[dim] = persistence_pairs
        # Store persistence points in PersistenceDiagram object
        diagram.persistence_pts = persistence_pts
        return diagram

    @staticmethod
    def _mask_infinities(array) -> np.ndarray:
        """
        Replaces infinite values in the persistence intervals with the maximum finite filtration value.

        Parameters
        ----------
        array : np.ndarray
            Array of persistence intervals.

        Returns
        -------
        np.ndarray
            Updated array where infinite values in the death time are replaced with the max finite value.
            If the array is empty or contains no finite death values, a small interval
            [[SMALLEST_NONZERO_VALUE, TINY_DEATH_THRESHOLD]] is returned.

        Notes
        -----
        SMALLEST_NONZERO_VALUE and TINY_DEATH_THRESHOLD are small constants defined to ensure
        numerical stability and prevent empty persistence intervals in extreme cases.
        """
        # Find the maximum finite value in the death times
        finite_values = array[array[:, 1] < np.inf, 1]
        if array.size == 0 or finite_values.size == 0:
            return np.array([SMALL_NONZERO_FEATURE])
        else:
            finite_max = np.max(finite_values)
            array[array[:, 1] == np.inf, 1] = finite_max + 1
            return array
