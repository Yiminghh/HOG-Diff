import pickle

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
import os
from utils import ho_utils, file_utils, graph_utils, data_utils, sbm_dataloader
from utils.data_utils import data_masker
from utils.path_manager import PathManager
import json
import logging
from tqdm import tqdm
from data import _GENERIC_DATASETS, _MOL_DATASETS
bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}


def process_mols(data_name, smile_list):
    if data_name in ['qm9']:
        max_node_num = 9
        atom_list = [6, 7, 8, 9]  # 6: C, 7: N, 8: O, 9: F
    elif data_name in ['zinc250k']:
        max_node_num = 38
        atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    elif data_name in ['guacamol']:
        max_node_num = 88
        atom_list = [6, 7, 8, 9, 5, 35, 17, 53, 15, 16, 34, 14] # C, N, O, F, B, Br, Cl, I, P, S, Se, Si
    elif data_name in ['moses']:
        max_node_num = 27
        atom_list = [6, 7, 16, 8, 9, 17, 35, 1] # # C, N, S, O, F, Cl, Br, H
    else:
        raise NotImplementedError


    atom_channel = len(atom_list)
    atom_index_map = {atomic_num: idx for idx, atomic_num in enumerate(atom_list)}

    xs, adjs = [], []
    current_max_mol_size = 0

    for smile in tqdm(smile_list, desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol)
        except Exception as e:
            continue

        num_atom = mol.GetNumAtoms()
        if num_atom > current_max_mol_size:
            current_max_mol_size = num_atom
            logging.info(f"Current max molecule size: {current_max_mol_size}")
        if num_atom > max_node_num:
            logging.warning(f"Molecule {smile} has {num_atom} atoms, which is greater than max_node_num {max_node_num}")
            continue

        # -------- process atoms --------
        atom_array = torch.zeros((max_node_num, atom_channel), dtype=torch.float)

        for atom_idx, atom in enumerate(mol.GetAtoms()):
            atom_feature = atom.GetAtomicNum()
            atom_array[atom_idx, atom_index_map[atom_feature]] = 1

        xs.append(atom_array)

        # -------- process bonds --------
        adj_array = torch.zeros([len(bond_type_to_int)+1, max_node_num, max_node_num], dtype=torch.float)
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            ch = bond_type_to_int[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj_array[ch, i, j] = 1.
            adj_array[ch, j, i] = 1.

        # The last channel stores the "no-bond" mask.
        adj_array[-1, :, :] = 1 - torch.sum(adj_array, dim=0)
        adjs.append(adj_array)

    x_tensor, adj_tensor = torch.stack(xs, dim=0), torch.stack(adjs, dim=0)

    adj_tensor = torch.argmax(adj_tensor, dim=1)
    adj_tensor = torch.where(adj_tensor == 3, 0, adj_tensor + 1).to(torch.float32)
    adj_tensor = adj_tensor / 3.0 # three kind of bonds

    return x_tensor, adj_tensor

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

def dataloader(config, mode='classical') -> DataLoader:
    data_name = config.data.name.lower()
    if data_name == 'sbm':
        return sbm_dataloader.sbm_dataloader(config, mode)
    elif data_name in _MOL_DATASETS:
        return mol_dataloader(config, mode)
    elif data_name in _GENERIC_DATASETS:
        return generic_dataloader(config, mode)

def mol_dataloader(config, mode='classical') -> DataLoader:
    dc = config.data
    data_name = dc.name.lower()
    latent_space = config.data.latent_space
    processed_dir = PathManager.DATA_PROCESSED_DIR
    raw_dir = PathManager.DATA_RAW_DIR
    processed_atom_bond_path = os.path.join(processed_dir, 'atom_bond.pt')
    if os.path.exists(processed_atom_bond_path) and not ('re_process' in config.data and config.data.re_process):
        (x_tensor, adj_tensor) = torch.load(processed_atom_bond_path)
        logging.info(f"Loaded processed atom bond tensor from {processed_atom_bond_path}")
    else:
        if data_name in ['qm9', 'zinc250k']:
            raw_file_path = os.path.join(raw_dir, f'{data_name}_property.csv')
            input_df = pd.read_csv(raw_file_path, sep=',', dtype='str')
            col = 'smile'
            smile_list = list(input_df[col])
            x_tensor, adj_tensor = process_mols(data_name, smile_list)

            valid_idx_path = os.path.join(raw_dir, f'valid_idx_{data_name}.json')
            with open(valid_idx_path) as f:
                test_idx = json.load(f)
            if data_name == 'qm9':
                test_idx = list(map(int, test_idx['valid_idxs']))

            train_idx = torch.tensor(list(set(range(len(x_tensor)))-set(test_idx)))
            x_tensor, adj_tensor = x_tensor[train_idx], adj_tensor[train_idx]
        elif data_name in ['guacamol','test']:
            raw_file_path = os.path.join(raw_dir, f'new_train.smiles')
            with open(raw_file_path, 'r') as f:
                smile_list = [line.strip() for line in f if line.strip()]
            x_tensor, adj_tensor = process_mols(data_name, smile_list)
        elif data_name in ['moses']:
            raw_file_path = os.path.join(raw_dir, f'train_moses.csv')
            input_df = pd.read_csv(raw_file_path, sep=',', dtype='str')
            col = 'SMILES'
            smile_list = list(input_df[col])
            x_tensor, adj_tensor = process_mols(data_name, smile_list)

        torch.save([x_tensor, adj_tensor], processed_atom_bond_path)
        logging.info(f"Saved processed atom bond tensor to {processed_atom_bond_path}")

    processed_laU_path = os.path.join(processed_dir, f'{latent_space}-laU.npy')
    if os.path.exists(processed_laU_path) and not ('re_process' in config.data and config.data.re_process):
        la, u = torch.load(processed_laU_path)
    else:
        la, u = transform_adjs(adj_tensor, latent_space)
        torch.save([la, u], processed_laU_path)



    if mode == 'classical':
        train_ds = TensorDataset(x_tensor, la, u, adj_tensor)
    elif mode in ['higher-order', 'OU']:
        lifting_type = dc.lifting_type
        min_media_size, max_media_size = dc.min_media_size, dc.max_media_size
        if lifting_type == 'SCs':
            assert min_media_size is not None, 'min_media_size must be provided for lifting_type SCs!'
            tag = min_media_size
        elif lifting_type == 'CCs':
            assert max_media_size is not None, 'max_media_size must be provided for lifting_type CCs!'
            tag = max_media_size
        ho_file_path = os.path.join(processed_dir, f'{data_name}-lifting-{lifting_type}({tag}).pt')


        if os.path.exists(ho_file_path) and not ('re_lifting' in dc and dc.re_lifting):
            (ho_x_tensor, ho_adj_tensor) = torch.load(ho_file_path)
        else:
            ho_adj_tensor = ho_utils.cell_complex_filter(adj_tensor, dc.max_node, max_media_size)
            ho_masker = data_masker(ho_adj_tensor, config)
            ho_x_tensor = ho_masker.mask_atom(x_tensor)

            torch.save([ho_x_tensor, ho_adj_tensor], ho_file_path)

        processed_ho_laU_path = os.path.join(processed_dir, f'{latent_space}-ho-laU.npy')
        if os.path.exists(processed_ho_laU_path) and not ('re_lifting' in dc and dc.re_lifting):
            (ho_la, ho_u) = torch.load(processed_ho_laU_path)
        else:
            ho_la, ho_u = transform_adjs(ho_adj_tensor, latent_space)
            torch.save([ho_la, ho_u], processed_ho_laU_path)


        if mode == 'higher-order':
            train_ds = TensorDataset(ho_x_tensor, ho_la, ho_u, ho_adj_tensor)
        elif mode == 'OU':
            train_ds = TensorDataset(x_tensor, la, u, adj_tensor,
                                     ho_x_tensor, ho_la, ho_u, ho_adj_tensor)
    logging.info("Finish loading data!")
    return train_ds

def calculate_avg_nodes_edges(adj_tensor):
    """
    Calculate the average number of nodes and edges in the networks.

    Parameters:
    adj_tensor (torch.Tensor): Adjacency tensor of shape (bs, n, n), where
                                bs is the batch size and n is the maximum number of nodes.

    Returns:
    tuple: (avg_nodes, avg_edges)
    """
    # Ensure the adjacency matrix is binary (0 or 1)
    adj_tensor = (adj_tensor > 0).float()

    # Calculate the number of nodes per graph (non-zero rows or columns)
    nodes_per_graph = (adj_tensor.sum(dim=-1) > 0).sum(dim=-1)

    # Calculate the number of edges per graph (sum of upper triangular elements)
    edges_per_graph = torch.triu(adj_tensor, diagonal=1).sum(dim=(-1, -2))

    # Calculate average number of nodes and edges
    avg_nodes = nodes_per_graph.float().mean().item()
    avg_edges = edges_per_graph.float().mean().item()

    return avg_nodes, avg_edges

def generic_dataloader(config, mode='classical') -> DataLoader:
    dc = config.data
    data_name = dc.name.lower()
    max_node_num = dc.max_node
    latent_space = config.data.latent_space
    pathloader = file_utils.PathLoader(config)


    with open(pathloader.raw_graph_path, 'rb') as f:
        pairwise_graph_list = pickle.load(f)
    test_size = int(dc.test_split * len(pairwise_graph_list))
    adj_tensor = graph_utils.nxgraphs2tensor(pairwise_graph_list[test_size:], max_node_num)
    x_tensor = graph_utils.init_features(dc, adj_tensor)
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
                ho_adj_tensor = ho_utils.simplicial_complex_filter(adj_tensor, dc.max_node, min_media_size, max_media_size)
            else:
                raise ValueError(f'lifting_type {lifting_type} not supported')
            torch.save(ho_adj_tensor, ho_file_path)

        ho_x_tensor = graph_utils.init_features(dc, ho_adj_tensor)
        ho_la, ho_u = transform_adjs(ho_adj_tensor, latent_space)


        if mode == 'higher-order':
            train_ds = TensorDataset(ho_x_tensor, ho_la, ho_u, ho_adj_tensor)
        elif mode == 'OU':
            train_ds = TensorDataset(x_tensor, la, u, adj_tensor,
                                     ho_x_tensor, ho_la, ho_u, ho_adj_tensor)

    return train_ds


@torch.no_grad()
def load_batch(graph_data, config, device, mode='classical'):
    """Extract features and masks from PyG Dense DataBatch.
    Returns:
        atom_feat: [bs, max_node, atom_channel]
        bond_feat: [bs, max_node, max_node]
    """
    scaler = data_utils.get_data_scaler(config)
    if mode in ['classical', 'higher-order']:
        x = graph_data[0].to(device)
        la = graph_data[1].to(device)
        u = graph_data[2].to(device)
        adj = graph_data[3].to(device)
        # When mode == 'higher-order', masker is actually a higher-order masker.
        masker = data_masker(adj, config)
    elif mode in ['OU']:
        # graph_data = [x, la, u, adj, ho_x, ho_la, ho_u, ho_adj]
        x = graph_data[0].to(device)
        la = graph_data[1].to(device)
        u = graph_data[2].to(device)
        adj = graph_data[3].to(device)
        ho_x = graph_data[4].to(device)
        ho_la = graph_data[5].to(device)
        ho_u = None  # ho_u is currently unused.
        ho_adj = graph_data[7].to(device)

        masker = data_masker(adj, config)
        ho_masker = data_masker(ho_adj, config)
        ho_x = ho_masker.mask_atom(scaler(ho_x, type='atom'))
        if config.data.name in _MOL_DATASETS:
            ho_x = ho_x + masker.mask_atom(torch.randn_like(ho_x, device='cpu').to(device))
        ho_la = ho_masker.mask_la(scaler(ho_la, type='eig'))
        ho_adj = ho_masker.mask_bond(scaler(ho_adj, type='bond'))

    x = masker.mask_atom(scaler(x, type='atom'))
    la = masker.mask_la(scaler(la, type='eig'))
    adj = masker.mask_bond(scaler(adj, type='bond'))
    if mode in ['classical', 'higher-order']:
        return x, la, u, adj, masker
    elif mode in ['OU']:
        return x, la, u, adj, ho_x, ho_la, ho_u, ho_adj, masker

