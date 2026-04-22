import pickle

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from utils import ho_utils, file_utils, graph_utils
import logging

def sbm_dataloader(config, mode='classical') -> DataLoader:



    dc = config.data
    data_name = dc.name.lower()
    max_node_num = dc.max_node
    latent_space = config.data.latent_space
    pathloader = file_utils.PathLoader(config)

    with open(pathloader.raw_graph_path, 'rb') as f:
        train_graphs, _, _ = pickle.load(f)
    # test_size = int(dc.test_split * len(pairwise_graph_list))
    adj_tensor = graph_utils.nxgraphs2tensor(train_graphs, max_node_num)
    x_tensor, feat_dim = init_features(dc, adj_tensor)
    la, u = transform_adjs(adj_tensor, latent_space)

    if mode == 'classical':
        train_ds = TensorDataset(x_tensor, la, u, adj_tensor)
    elif mode in ['higher-order', 'OU']:
        lifting_type = dc.lifting_type
        min_media_size, max_media_size = dc.min_media_size, dc.max_media_size
        ho_file_path = pathloader.get_ho_file_path(lifting_type, min_media_size, max_media_size)
        if os.path.exists(ho_file_path) and not ('re_lifting' in dc and dc.re_lifting):
            ho_adj_tensor = torch.load(ho_file_path)
        else:
            if lifting_type == 'CCs':
                ho_adj_tensor = ho_utils.cell_complex_filter(adj_tensor, dc.max_node, max_media_size)
            elif lifting_type == 'SCs':
                ho_adj_tensor = ho_utils.simplicial_complex_filter(adj_tensor, dc.max_node, min_media_size,
                                                                   max_media_size)
            else:
                raise ValueError(f'lifting_type {lifting_type} not supported')
            torch.save(ho_adj_tensor, ho_file_path)

        ho_x_tensor, _ = init_features(dc, ho_adj_tensor)
        ho_la, ho_u = transform_adjs(ho_adj_tensor, latent_space)

        if mode == 'higher-order':
            train_ds = TensorDataset(ho_x_tensor, ho_la, ho_u, ho_adj_tensor)
        elif mode == 'OU':
            train_ds = TensorDataset(x_tensor, la, u, adj_tensor,
                                     ho_x_tensor, ho_la, ho_u, ho_adj_tensor)

    return train_ds


def init_features(config_data, adjs):
    eps = 1e-5
    flags = torch.abs(adjs).sum(-1).gt(eps).to(dtype=torch.float32)
    if len(flags.shape) == 3:
        flags = flags[:, 0, :]

    feature = []
    feat_dim = []
    for feat_type in config_data.feat.type:
        if feat_type == 'deg':
            # deg = adjs.sum(dim=-1).to(torch.long)
            # feat = F.one_hot(deg, num_classes=config_data.max_feat_num).to(torch.float32)
            feat = adjs.sum(dim=-1).unsqueeze(-1).to(torch.float32) / config_data.feat.scale
        elif 'eig' in feat_type:
            idx = int(feat_type.split('eig')[-1])
            eigvec = EigenFeatures(idx)(adjs, flags)
            feat = eigvec[..., -1:] * config_data.feat.scale
        else:
            raise NotImplementedError(f'Feature: {feat_type} not implemented.')
        feature.append(feat)
        feat_dim.append(feat.shape[-1])
    feature = torch.cat(feature, dim=-1) * flags[:, :, None]

    return feature, feat_dim


def transform_adjs(adjs_tensor, latent_space='Laplacian'):
    if latent_space == 'Laplacian':
        D = torch.sum(adjs_tensor, dim=-1)
        L = torch.diag_embed(D) - adjs_tensor
    elif latent_space == 'adj':
         L = adjs_tensor
    else:
        raise NotImplementedError

    try:
        la, u = torch.linalg.eigh(L)
    except Exception as e:
        logging.warning(f"Error occurs when performing torch.linalg.eigh: {e}")
        la, u = np.linalg.eigh(L.cpu().numpy())
        la, u = torch.tensor(la, device=L.device), torch.tensor(u, device=L.device)

    # u @ torch.diag_embed(la) @ (u.transpose(-1, 2))
    if latent_space == 'Laplacian':
        sorted_la = torch.flip(la, dims=[1])
        sorted_u = torch.flip(u, dims=[2])
    elif latent_space == 'adj':
        sorted_la, sorted_u = la, u

    # torch.max(sorted_u @ torch.diag_embed(sorted_la) @ sorted_u.transpose(-2, -1) - L)
    return sorted_la, sorted_u


### code adapted from https://github.com/cvignac/DiGress/blob/main/src/diffusion/extra_features.py
class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, k=2):
        self.num_eig_vec = k

    def __call__(self, adj, mask):
        mask = mask.to(torch.long).bool()
        A = adj.float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

        # Retrieve eigenvectors features
        k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                            node_mask=mask,
                                                            n_connected=n_connected_comp,
                                                            k=self.num_eig_vec)
        return k_lowest_eigenvector


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2

def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev <= 1e-5).sum(dim=-1)
    #assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    Warning: this function does not exactly return what is desired, the lcc might not be exactly the returned vector.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return first_k_ev
