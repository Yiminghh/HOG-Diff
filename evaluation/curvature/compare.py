import networkx as nx
from .kilt import KILT
from .topology.distances import (
    TopologicalDistance,
    supported_distances,
)
from typing import List, Optional
from joblib import Parallel, delayed

class Comparator:
    """Compare Graphs or Graph Distributions using Curvature Filtrations as in https://openreview.net/forum?id=Dt71xKyabn using `KILT`.

    Initialization of object allows user to specify parameters to customize how curvature is calculated.

    Attributes
    ----------
    kilt : KILT
        The KILT object facilitates computing discrete curvature values for graphs.
    homology_dims : List[int]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1). Default is [0, 1], the usual choice for graphs.

    extended_persistence : bool
        If True, the extended persistence diagram is computed. Default is False.

    descriptor1 : One of {PersistenceLandscape (default), PersistenceImage}
        The summarizing topological descriptor for G1, which encodes persistent homology information.
    descriptor2 : One of {PersistenceLandscape (default), PersistenceImage}
        The summarizing topological descriptor for G2.
        Must match type of descriptor1 in order to compute distance, which encodes persistent homology inf`ormation.
    """

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
        homology_dims: Optional[List[int]] = [0, 1],
        extended_persistence: bool = False,
        n_jobs: int = 1,          # parallel worker count; 1 disables parallelism.
    ) -> None:
        """
        Parameters
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
            Used only if Ollivier--Ricci curvature is calculated, with default None. If set, should refer to a function that calculate a probability measure for a given graph and a given node. This callable needs to satisfy the following signature: ``prob_fn(G, node, node_to_index)``
            Here, `G` refers to the graph, `node` to the node whose measure is to be calculated, and `node_to_index` to the lookup map that maps a node identifier to a zero-based index.
            If `prob_fn` is set, providing `alpha` will not have an effect.

        homology_dims : List[int]
            Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
            Default is [0, 1].

        extended_persistence : bool
            If True, the extended persistence diagram is computed. Default is False.

        Returns
        -------
        None
        """

        # Initialize Curvature and GraphHomology objects
        self.kilt = KILT(measure, weight, alpha, prob_fn)
        self.homology_dims = homology_dims
        self.extended_persistence = extended_persistence

        self.descriptor1 = None
        self.descriptor2 = None
        self.n_jobs = n_jobs

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Key Members: Fit and Transform                           │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, G1, G2, metric="landscape", **kwargs) -> None:
        """Initializes a Topological Distance object and computes the topological descriptors that will be compared.
        Stores these topological descriptors in the descriptor1 and descriptor2 attributes of the Comparator object.

        Parameters
        ----------
        G1 : nx.Graph or List[nx.Graph]
            The first graph or graph distribution for comparison.
        G2 : nx.Graph or List[nx.Graph]
            The first graph or graph distribution for comparison.
        metric : str, default="landscape"
            One of: {"landscape", "image"}. Indicates which topological descriptor to use for computing distance.

        Returns
        -------
        None
        """
        cls = self._setup_distance(metric)

        pd_1 = self._curvature_filtration(G1)
        pd_2 = self._curvature_filtration(G2)

        self.distance = cls(
            diagram1=pd_1,
            diagram2=pd_2,
            **kwargs,
        )  # This should error if theres no distribution support

        self.descriptor1, self.descriptor2 = self.distance.fit()

    def transform(self) -> float:
        """Computes the numeric distance between topological descriptors, i.e. attributes self.descriptor1 and self.descriptor2.
        Can only be run after fit().

        Returns
        -------
        float :
            The distance between the topological descriptors stored in the descriptor1 and descriptor2 attributes.

        """
        assert (
            self.descriptor1 and self.descriptor2
        ), "You have not fitted topological descriptors. Try running `fit` or `fit_transform`"  # check that these are not None

        return self.distance.transform(self.descriptor1, self.descriptor2)

    def fit_transform(self, G1, G2, metric="landscape", **kwargs) -> float:
        """Runs the fit() and transform() methods in succession.
        Returns a numeric distance between G1 and G2, computed according to the given metric.

        Parameters
        ----------
        G1 : nx.Graph or List[nx.Graph]
            The first graph or graph distribution for comparison.
        G2 : nx.Graph or List[nx.Graph]
            The first graph or graph distribution for comparison.
        metric : str, default="landscape"
            One of: {"landscape", "image"}. Indicates which topological descriptor to use for computing distance.

        Returns
        -------
        float :
            The distance between G1 and G2.


        Example
        -------
        >>> import networkx as nx
        >>> from scott.compare import Comparator
        >>> graph1 = nx.random_geometric_graph(100, 0.1)
        >>> graph2 = nx.random_geometric_graph(100, 0.2)
        >>> comp = Comparator(measure='ollivier_ricci_curvature')
        >>> distance = comp.fit_transform(graph1, graph2, metric='image')
        >>> print(distance)
        >>> 9.274264343274613
        """
        self.fit(G1, G2, metric, **kwargs)
        return self.transform()

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _setup_distance(self, metric) -> TopologicalDistance:
        """Checks that metric is supported. If so, returns TopologicalDistance subclass associated with the metric."""
        # Check that the metric is supported
        assert metric in supported_distances, f"Metric {metric} is not supported."

        return supported_distances[metric]

    def _curvature_filtration(self, G):
        """Computes a curvature filtration for all graphs in G, returning a list of PersistenceDiagrams."""
        graph_iterable = self._format_inputs(G)
        if self.n_jobs <=1:
            return [self._kilterator(g) for g in graph_iterable]
        
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._kilterator)(g) for g in graph_iterable
        )

    def _kilterator(self, graph):
        """Computes a curvature filtration for one graph, returning a PersistenceDiagram."""
        return self.kilt.fit_transform(
            graph,
            homology_dims=self.homology_dims,
            extended_persistence=self.extended_persistence,
            mask_infinite_features=True,
        )

    @staticmethod
    def _format_inputs(G):
        """Always returns a list of graphs."""
        if Comparator._is_distribution(G):
            return G
        elif Comparator._is_graph(G):
            return [G]
        else:
            raise ValueError("Input must be a networkx.Graph or a list of networkx.Graphs")

    @staticmethod
    def _is_distribution(G):
        """Returns True if G is a list of graphs, False otherwise."""
        return isinstance(G, list)

    @staticmethod
    def _is_graph(G):
        """Returns True if G is a single graph, False otherwise"""
        return isinstance(G, nx.Graph)

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the Comparator object."""
        name = f"Comparator object with: \n\tCurvature method defined by: [{self.kilt}]\n\tDescriptor 1: {self.descriptor1}\n\tDescriptor 2: {self.descriptor2}"
        return name

    def __repr__(self) -> str:
        """Returns a string representation of the Comparator object helpful to the developer."""
        return f"Comparator({self.kilt}, {self.descriptor1}, {self.descriptor2}, {self.homology_dims}, {self.extended_persistence})"
