
import torch

import networkx as nx
from tqdm import tqdm
import itertools


def cell_complex_filter(graph_tensor, max_node_num=None, max_media_size=10, show_progress=True) -> torch.Tensor:
    """Cell complex filtering (CCF) from HOG-Diff (arXiv:2502.04308, Prop. 2).

    Keeps only edges lying on a cycle of length ≤ ``max_media_size`` — i.e.
    edges that belong to some 2-cell of the lifted cell complex. Implemented by
    deleting each edge in turn and probing reachability via matrix powers.

    Args:
        graph_tensor: adjacency tensor of shape [N, N] (single graph) or [B, N, N] (batch).
        max_node_num: pad each output to this size. Defaults to N (no padding).
        max_media_size: maximum cycle length considered.
        show_progress: whether to display a tqdm progress bar.

    Returns:
        Filtered adjacency tensor of the same rank as the input.

    Example:
        >>> adj = torch.tensor([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=torch.float)
        >>> cell_complex_filter(adj, max_media_size=3)  # edge (2,3) dropped: no cycle through it
    """
    single = graph_tensor.dim() == 2
    if single:
        graph_tensor = graph_tensor.unsqueeze(0)
    if max_node_num is None:
        max_node_num = graph_tensor.shape[-1]

    filtered_graph_list = []
    for adj_tensor in tqdm(graph_tensor, total=len(graph_tensor),
                           desc='filtering graphs ...', colour='green',
                           disable=not show_progress):
        A = adj_tensor.numpy()
        G = nx.from_numpy_array(A)
        # networkx node labels are not guaranteed to be 0..N-1, so remap.
        node_id = {node: id for id, node in enumerate(G.nodes())}

        ho_adj_tensor = torch.zeros(max_node_num, max_node_num)
        for (v1, v2) in G.edges():
            v1_id, v2_id = node_id[v1], node_id[v2]
            tA = A.copy()
            tA[v1_id, v2_id] = tA[v2_id, v1_id] = 0
            A_power = tA.copy()
            for _ in range(max_media_size - 2):
                A_power = A_power @ tA
                if A_power[v1_id, v2_id] > 0:
                    ho_adj_tensor[v1_id, v2_id] = ho_adj_tensor[v2_id, v1_id] = adj_tensor[v1_id, v2_id]
                    break
        filtered_graph_list.append(ho_adj_tensor)
    out = torch.stack(filtered_graph_list)
    return out[0] if single else out


def simplicial_complex_filter(graph_tensor, max_node_num=None, min_size=3, max_size=10, show_progress=True) -> torch.Tensor:
    """Simplicial complex filtering from HOG-Diff (arXiv:2502.04308, Prop. 2).

    Keeps only edges that participate in a clique of size in [``min_size``, ``max_size``],
    i.e. edges spanning a (p-1)-simplex with p ≥ ``min_size`` of the lifted simplicial complex.

    Args:
        graph_tensor: adjacency tensor of shape [N, N] (single graph) or [B, N, N] (batch).
        max_node_num: pad each output to this size. Defaults to N (no padding).
        min_size: minimum clique size kept (3 = triangles).
        max_size: maximum clique size enumerated.
        show_progress: whether to display a tqdm progress bar.

    Returns:
        Filtered adjacency tensor of the same rank as the input.
    """
    single = graph_tensor.dim() == 2
    if single:
        graph_tensor = graph_tensor.unsqueeze(0)
    if max_node_num is None:
        max_node_num = graph_tensor.shape[-1]

    filtered_graph_list = []
    for adj_tensor in tqdm(graph_tensor, total=len(graph_tensor),
                           desc='filtering graphs ...', colour='green',
                           disable=not show_progress):
        G = nx.from_numpy_array(adj_tensor.numpy())

        ho_adj_tensor = torch.zeros(max_node_num, max_node_num)
        for clique in nx.enumerate_all_cliques(G):
            if len(clique) < min_size:
                continue
            for v1, v2 in itertools.combinations(clique, 2):
                ho_adj_tensor[v1, v2] = ho_adj_tensor[v2, v1] = adj_tensor[v1, v2]
            if len(clique) > max_size:
                break
        filtered_graph_list.append(ho_adj_tensor)

    out = torch.stack(filtered_graph_list)
    return out[0] if single else out
