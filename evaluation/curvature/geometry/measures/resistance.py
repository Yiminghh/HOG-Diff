import warnings
import networkx as nx
import numpy as np
from joblib import Parallel, delayed


#  ╭──────────────────────────────────────────────────────────╮
#  │ Resistance                                               │
#  ╰──────────────────────────────────────────────────────────╯


def resistance_curvature(G, weight=None, n_jobs=-1):
    """Calculate Resistance Curvature of a graph.

    This function calculates the resistance curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores.
        Set to 1 for sequential processing. Default is -1.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # Generate Matrix of Resistance Distances and Node Reference
        # Dictionary
        R, node_to_index = _pairwise_resistances(G, weight=weight)

    # Convert to list for parallel processing
    edges = list(G.edges())

    # Parallel computation
    curvature_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_edge_resistance_curvature)(G, edge, weight, R, node_to_index)
        for edge in edges
    )

    return np.asarray(curvature_values)


def _compute_single_edge_resistance_curvature(G, edge, weight, R, node_to_index):
    """Compute resistance curvature for a single edge (for parallel execution)."""
    source, target = edge
    source_curvature = _node_resistance_curvature(
        G, source, weight=weight, R=R, node_to_index=node_to_index
    )
    target_curvature = _node_resistance_curvature(
        G, target, weight=weight, R=R, node_to_index=node_to_index
    )

    edge_curvature = (
        2 * (source_curvature + target_curvature) / R[node_to_index[source], node_to_index[target]]
    )
    return edge_curvature


def _pairwise_resistances(G, weight=None):
    """Calculate pairwise resistances for all neighbors of a graph.

    Calculate pairwise resistances for all neighbors in a graph `G`
    using the networkx implementation of `resistance_distance`. This
    function helps reducing redundant computations when calculating
    `resistance_curvature`, by doing the necessary calculations up
    front.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    R : np.matrix
        A matrix of pairwise resistance distances between neighboring
        nodes in `G`.

    node_to_index : dict
        A reference dictionary for translating between nodes and indices
        of `G`.
    """
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    n = len(G.nodes())

    # Initialize nxn Matrix
    R = np.zeros(shape=(n, n))

    # List of connected components with original node order
    components = list([G.subgraph(c).copy() for c in nx.connected_components(G)])
    for C in components:
        for source, target in C.edges():
            i, j = node_to_index[source], node_to_index[target]
            r = nx.resistance_distance(
                C,
                source,
                target,
                weight=weight,
                invert_weight=False,
            )
            # Assign Matrix Entries for neighbors
            R[i, j], R[j, i] = r, r

    return R, node_to_index


def _node_resistance_curvature(G, node, weight=None, R=None, node_to_index=None):
    """Calculate Resistance Curvature of a given node in a graph 'G'.

    This function calculates the resistance curvature of only
    the nodes in a graph, optionally takes (positive)
    edge weights into account. This is a helper function for
    resistance_curvature; the curvature of each node is used to
    determine the overall curvature of the graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    np.float32
        The node curvature of a given node in `G`.
    """
    assert node in G.nodes()

    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    if R is None:
        R, node_to_index = _pairwise_resistances(G, weight=weight)

    neighbors = list(G.neighbors(node))
    rel_resistance = 0

    for neighbor in neighbors:

        if weight is not None and len(G.get_edge_data(node, neighbor)) > 0:
            w = G[node][neighbor][weight]

        else:
            w, G[node][neighbor]["weight"] = 1, 1

        rel_resistance += R[node_to_index[node]][node_to_index[neighbor]] * w

    node_curvature = 1 - 0.5 * rel_resistance

    return np.float32(node_curvature)
