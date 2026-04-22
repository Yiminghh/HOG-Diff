import networkx as nx
import numpy as np
from joblib import Parallel, delayed


#  ╭──────────────────────────────────────────────────────────╮
#  │ Forman-Ricci                                             │
#  ╰──────────────────────────────────────────────────────────╯


def forman_curvature(G, weight=None, n_jobs=-1):
    """Calculate Forman--Ricci curvature of a graph.

    This function calculates the Forman--Ricci curvature of a graph,
    optionally taking (positive) node and edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. Will use the same attribute to look up node weights. If
        None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    # This calculation is much more efficient than the weighted one, so
    # we default to it in case there are no weights in the graph.
    if weight is None:
        return _forman_curvature_unweighted(G, n_jobs)
    else:
        return _forman_curvature_weighted(G, weight, n_jobs)


def _forman_curvature_unweighted(G, n_jobs):
    # Convert to list for parallel processing
    edges = list(G.edges())

    # Parallel computation
    curvature_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_edge_forman_unweighted)(G, edge) for edge in edges
    )

    return np.asarray(curvature_values)


def _forman_curvature_weighted(G, weight, n_jobs):
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    # Convert to list for parallel processing
    edges = list(G.edges())

    # Parallel computation
    curvature_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_edge_forman_weighted)(G, edge, weight, has_node_attributes)
        for edge in edges
    )

    return np.asarray(curvature_values)


def _compute_single_edge_forman_unweighted(G, edge):
    """Compute Forman curvature for a single edge in unweighted graph (for parallel execution)."""
    source, target = edge
    source_degree = G.degree(source)
    target_degree = G.degree(target)

    source_neighbours = set(G.neighbors(source))
    target_neighbours = set(G.neighbors(target))

    n_triangles = len(source_neighbours.intersection(target_neighbours))
    return float(4 - source_degree - target_degree + 3 * n_triangles)


def _compute_single_edge_forman_weighted(G, edge, weight, has_node_attributes):
    """Compute Forman curvature for a single edge in weighted graph (for parallel execution)."""
    source, target = edge
    source_weight, target_weight = 1.0, 1.0

    # Makes checking for duplicate edges easier below. We expect the
    # source vertex to be the (lexicographically) smaller one.
    # Use string representation for comparison to handle mixed node types
    if str(source) > str(target):
        source, target = target, source

    if has_node_attributes:
        source_weight = G.nodes[source][weight]
        target_weight = G.nodes[target][weight]

    edge_weight = G[source][target][weight]

    e_curvature = source_weight / edge_weight
    e_curvature += target_weight / edge_weight

    parallel_edges = list(G.edges(source, data=weight)) + list(G.edges(target, data=weight))

    for u, v, w in parallel_edges:
        # Use string representation for comparison to handle mixed node types
        if str(u) > str(v):
            u, v = v, u

        if (u, v) == edge:
            continue
        else:
            e_curvature -= w / np.sqrt(edge_weight * w)

    e_curvature *= edge_weight
    return float(e_curvature)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Balanced Forman                                          │
#  ╰──────────────────────────────────────────────────────────╯


def balanced_forman_curvature(G, weight=None, n_jobs=-1):
    """
    Compute the balanced Forman curvature for each edge in a NetworkX graph.
    Defaults to parallel processing.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (weighted or unweighted).
    weight : str or None
        Name of the edge weight attribute. If None, the graph is treated as unweighted.
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores.
        Set to 1 for sequential processing. Default is -1.

    Returns
    -------
    np.array
        An array of balanced Forman curvature values, following the ordering of edges of `G`.

    References
    ----------
    Topping, Jake, et al. "Understanding Over-Squashing and Bottlenecks on Graphs via Curvature." International Conference on Learning Representations. 2022.
    """
    A, d_in, d_out, node_to_index = _prepare_graph_data(G, weight)
    A_squared = np.matmul(A, A)

    # Convert to list for parallel processing
    edges = list(G.edges())

    # Parallel computation
    curvature_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_edge_curvature)(A, A_squared, d_in, d_out, node_to_index, u, v)
        for u, v in edges
    )

    return np.asarray(curvature_values)


def _compute_single_edge_curvature(A, A_squared, d_in, d_out, node_to_index, u, v):
    """Compute curvature for a single edge (for parallel execution)."""
    i, j = node_to_index[u], node_to_index[v]
    weight = A[i, j]

    if weight == 0:
        return 0

    d_max = max(d_in[i], d_out[j])
    d_min = min(d_in[i], d_out[j])

    if d_max * d_min == 0:
        return 0

    sharp_ij, lambda_ij = _compute_sharpness_and_lambda_optimized(A, A_squared, i, j, weight)

    curvature = (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A_squared[i, j] * weight

    if lambda_ij > 0:
        curvature += sharp_ij / (d_max * lambda_ij)

    return curvature


def _prepare_graph_data(G, weight):
    """Prepare weighted adjacency matrix and degree information."""
    A = nx.to_numpy_array(G, weight=weight)
    d_in = A.sum(axis=0)  # Weighted in-degrees
    d_out = A.sum(axis=1)  # Weighted out-degrees

    # Create node to index mapping
    node_to_index = {node: i for i, node in enumerate(G.nodes())}

    return A, d_in, d_out, node_to_index


def _compute_sharpness_and_lambda_optimized(A, A_squared, i, j, weight):
    """Optimized sharpness computation using precomputed A_squared."""
    sharp_ij = 0
    lambda_ij = 0

    # Vectorized computation over all k
    TMP_1 = A[:, j] * (A_squared[i, :] - A[i, :]) * weight
    TMP_2 = A[i, :] * (A_squared[:, j] - A[:, j]) * weight

    # Count positive values and find max
    positive_1 = TMP_1 > 0
    positive_2 = TMP_2 > 0

    sharp_ij = np.sum(positive_1) + np.sum(positive_2)

    max_1 = np.max(TMP_1) if np.any(positive_1) else 0
    max_2 = np.max(TMP_2) if np.any(positive_2) else 0
    lambda_ij = max(max_1, max_2)

    return sharp_ij, lambda_ij
