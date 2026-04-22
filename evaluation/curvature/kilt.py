import networkx as nx
import numpy as np
from typing import List, Optional


from .geometry import measures
from .topology.representations import PersistenceDiagram
from .topology.ph import GraphHomology

# methods for calculating curvature that KILT currently supports
CURVATURE_MEASURES = [
    "forman_curvature",
    "ollivier_ricci_curvature",
    "resistance_curvature",
    "balanced_forman_curvature",
]


class KILT:
    """Krvature-Informed Links and Topology (KILT) is a class that faciltates computing discrete curvature values for graphs.
     Curvature values can be used as a filtration for persistent homology (as done in https://openreview.net/forum?id=Dt71xKyabn), providing even more expressive descriptors for the structure of a graph.

    Attributes
    ----------
    measure : string, default= "forman_curvature"
        The type of curvature measure to be calculated. See CURVATURE_MEASURES for the methods that KILT functionality currently supports.

    alpha : float, default=0.0
        Only used if Olivier--Ricci curvature is calculated, with default 0.0. Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None, default=None
        Can be specified for (weighted) Forman--Ricci, Olivier--Ricci and Resistance curvature measures. Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that for Ollivier--Ricci curvature, if `prob_fn` is provided, this parameter will have no effect for the calculation of probability measures, but it will be used for the calculation of shortest-path distances.

    prob_fn : callable or None, default=None
        Used only if Ollivier--Ricci curvature is calculated, with default None. If set, should refer to a function that calculate a probability measure for a given graph and a given node. This callable needs to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure is to be calculated, and `node_to_index` to the lookup map that maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    n_jobs : int, default=-1
        Number of parallel jobs for curvature computation. -1 means use all available cores.
        Set to 1 for sequential processing. Applies to all curvature measures that support parallel processing.
    """

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
        n_jobs=-1,
    ) -> None:
        """Creates an instance of the KILT class with the desired specifications for the method of computing curvature in a graph."""

        # Check that curvature method is supported
        assert (
            measure in CURVATURE_MEASURES
        ), "The given curvature measure is not yet supported by KILT."

        # Assign curvature attributes
        self.measure = measure
        self.weight = weight
        self.alpha = alpha
        self.prob_fn = prob_fn
        self.n_jobs = n_jobs

        # Initialize graph
        self._G = None

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Getters and Setters                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    @property
    def G(self) -> nx.Graph:
        """Getter method for the graph associated with the KILT object."""
        return self._G

    @G.setter
    def G(self, graph) -> None:
        """Setter method which sets the given nx.Graph as the KILT object's graph attribute."""
        self._G = graph.copy()

    @property
    def curvature(self) -> Optional[np.ndarray]:
        """Getter method for the curvature values of the graph."""
        if self.G is None:
            return None
        return self._unpack_curvature_values(nx.get_edge_attributes(self.G, name=self.measure))

    @curvature.setter
    def curvature(self, values: list) -> None:
        """Setter method for the curvature values of the graph."""
        edge_map = self._pack_curvature_values(self.G.edges, values)
        nx.set_edge_attributes(self.G, edge_map, name=self.measure)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Geometry                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, graph) -> None:
        """
        Calculates the curvature values for the edges of the given graph according to the specifications of the KILT object
        Curvature values are can then be retrieved via the curvature attribute.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph for which curvature values will be calculated.

        Returns
        -------
        None

        Examples
        --------
        >>> import networkx as nx
        >>> from scott.topology.kilt import KILT
        >>> G = nx.Graph()
        >>> edges = [(0, 1),(0, 2),(0, 3),(0, 4),(1, 2),(2, 3),(3, 5),(4, 5),(5, 6),(3, 7)]
        >>> kilt = KILT()
        >>> kilt.fit(G)
        >>> print(kilt.curvature)
        [ 1.  3. -1. -2.  2.  0. -3. -1. -1.  0.]
        """
        self._graph_is_not_empty(graph)
        self.G = graph
        edge_curvatures = self._compute_curvature(graph)
        self.curvature = edge_curvatures

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Topology                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def transform(
        self,
        homology_dims: Optional[List[int]] = [
            0,
            1,
        ],
        mask_infinite_features: bool = False,
        extended_persistence: bool = True,
    ) -> PersistenceDiagram:
        """
        Executes a curvature filtration for the given homology dimensions.
        Can only be run after fit(), as it requires edge curvature values to execute a filtration.

        Parameters
        ----------
        homology_dims : List[int]
            A list of the homology dimensions for which persistence points should be calculated, with default being [0,1].

        Returns
        -------
        PersistenceDiagram (Dict[int, np.array])
            A persistence diagram wrapper for the topological information resulting from a curvature filtration.
            Attribute persistence_pts stores a Dict[int, np.array] that a maps homology dimension key to a np.array of its persistence pairs.
        """
        ph = GraphHomology(
            homology_dims,
            self.measure,
            mask_infinite_features=mask_infinite_features,
            extended_persistence=extended_persistence,
        )
        assert (
            self._curvature_values_exist()
        ), "Curvature values have not been computed. Please run `fit` first."
        return ph.calculate_persistent_homology(self.G)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Both: A Curvature Filtration!                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit_transform(
        self,
        graph,
        homology_dims: Optional[List[int]] = [
            0,
            1,
        ],
        mask_infinite_features: bool = False,
        extended_persistence: bool = False,
    ) -> PersistenceDiagram:
        """Runs fit(graph) and transform(homology_dims) in succession.
        Thus computes the curvature values for the given graph and executes a filtration for the given homology dimensions.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph for which curvature values will be calculated.
        homology_dims : List[int]
            A list of the homology dimensions for which persistence points should be calculated, with default being [0,1].
        mask_infinite_features : bool, default=False
            Whether to mask infinite persistence pairs in the resulting persistence diagrams.
        extended_persistence : bool, default=False
            Whether to compute extended persistence. Default is False, computing standard persistence.

        Returns
        -------
        PersistenceDiagram (Dict[int, np.array])
            A persistence diagram wrapper for the topological information resulting from a curvature filtration.
            Attribute persistence_pts stores a Dict[int, np.array] that a maps homology dimension key to a np.array of its persistence pairs.

        Examples
        --------
        >>> import networkx as nx
        >>> from scott.topology.kilt import KILT
        >>> G = nx.Graph()
        >>> edges = [(0, 1),(0, 2),(0, 3),(0, 4),(1, 2),(2, 3),(3, 5),(4, 5),(5, 6),(3, 7)]
        >>> G.add_edges_from(edges)
        >>> kilt = KILT()
        >>> diagram = kilt.fit_transform(G,mask_infinite_features=False,extended_persistence=False)
        >>> print(diagram.persistence_pts)
        {0: array([[-2., -1.],
           [-3., inf]]), 1: array([[ 2.,  3.],
           [-1., inf]])}
        """
        self.fit(graph)
        return self.transform(
            homology_dims,
            mask_infinite_features,
            extended_persistence,
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _compute_curvature(self, graph) -> np.ndarray:
        """Helper function for calculating the edge curvature values for the given graph according to the specifications of the KILT object (i.e. measure, weight, etc.)."""
        curvature_fn = getattr(measures, self.measure)
        if self.measure == "ollivier_ricci_curvature":
            # Ollivier Ricci measure supports extra inputs
            return curvature_fn(
                graph,
                self.alpha,
                self.weight,
                prob_fn=self.prob_fn,
                n_jobs=self.n_jobs,
            )
        else:
            # Forman and Resistance measures only require graph and optional weight
            return curvature_fn(graph, self.weight, n_jobs=self.n_jobs)

    def _curvature_values_exist(self) -> bool:
        """Checks whether curvature values have already been computed and stored in the KILT object's curvature attribute."""
        return self.curvature is not None and len(self.curvature) > 0

    def _graph_is_not_empty(self, graph) -> None:
        """Asserts that the inputted graph is not empty (i.e. has nodes and edges)."""
        assert len(graph.nodes) > 0, "Graph must have nodes to compute curvature"

    @staticmethod
    def _unpack_curvature_values(
        dict,
    ) -> np.ndarray:
        """Converts the curvature values from a dictionary to a numpy array."""
        return np.array(list(dict.values()))

    @staticmethod
    def _pack_curvature_values(edges, values) -> dict:
        """Converts corresponding lists of edges and curvature values into a dictionary mapping edges to their curvature value."""
        return dict(zip(edges, values))

    def __str__(self) -> str:
        """Return a string representation of the KILT object and any custom attributes."""
        name = f"KILT Object with Measure: {self.measure}, Graph: {self.G}"
        if self.weight != None:
            name += f", Custom Weight Attribute: {self.weight}"
        if self.alpha != 0.0:
            name += f", Custom Alpha: {self.alpha}"
        if self.prob_fn != None:
            name += f", Custom Probability Function: {self.prob_fn}"
        return name

    def __repr__(self) -> str:
        """Return a string representation of the KILT object."""
        return f"KILT({self.measure}, {self.weight}, {self.alpha}, {self.prob_fn}, {self.n_jobs})"
