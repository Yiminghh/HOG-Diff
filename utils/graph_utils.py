import networkx as nx
import torch
def nxgraphs2tensor(graph_list:nx.Graph, max_node_num):
    """
    Convert a list of nx.Graph into a tensor representation.
    Parameters:
    - max_node_num (int): The maximum number of nodes in the graphs. Pads the matrices to ensure they have uniform dimensions (`max_node_num` x `max_node_num`).
    """
    adjs_list = []

    def pad_adjs(ori_adj, node_number):
        a = torch.tensor(ori_adj)
        ori_len = a.shape[-1]
        if ori_len == node_number:
            return a
        if ori_len > node_number:
            raise ValueError(f'ori_len {ori_len} > node_number {node_number}')

        # Allocate a zero matrix at the target size and copy in the original adjacency.
        padded_adj = torch.zeros((node_number, node_number))
        padded_adj[:ori_len, :ori_len] = a
        return padded_adj

    for g in graph_list:
        adj = nx.to_numpy_array(g)

        padded_adj = pad_adjs(adj, node_number=max_node_num).to(dtype=torch.float32)
        adjs_list.append(padded_adj)


    adjs_tensor = torch.stack(adjs_list)
    return adjs_tensor

def tensor2nxgraphs(adjs_tensor, thr=0.5):

    # quantize
    adjs_tensor = torch.where(adjs_tensor < thr, torch.zeros_like(adjs_tensor), torch.ones_like(adjs_tensor))

    graph_list = []
    for adj in adjs_tensor:

        adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(adj)
        # G = nx.Graph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list



def init_features(data_config, adj_tensor):

    bs, N, _ = adj_tensor.size()
    feature_keys = data_config.features
    valid_features = ['deg', 'eig']
    assert set(feature_keys).issubset(set(valid_features)), f"{feature_keys} not in {valid_features}"

    features = []

    if 'deg' in feature_keys:
        features.append(adj_tensor.sum(dim=-1).unsqueeze(-1).to(torch.float32))
    if 'eig' in feature_keys:
        _, u0 = torch.linalg.eigh(adj_tensor)
        features.append(u0.to(torch.float32))

    return torch.cat(features, dim=-1)

