"""
analysis.sbm_evaluator
from analysis.sbm_evaluator import eval_VUN, eval_sbm_validity, is_sbm_graph
NOTE: import this module early — graph_tool can conflict with other libraries otherwise.
"""

import os
import sys
import pickle
import concurrent.futures

import graph_tool.all as gt
import networkx as nx
import numpy as np
from scipy.stats import chi2


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        return 0.0

    for _ in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9
    return p


def eval_sbm_validity(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for _ in range(len(G_list))],
                                     [p_inter for _ in range(len(G_list))],
                                     [strict for _ in range(len(G_list))],
                                     [refinement_steps for _ in range(len(G_list))]):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    """Fraction of ``fake_graphs`` isomorphic to any ``train_graph`` (used to compute novelty)."""
    if len(fake_graphs) == 0:
        return 0.0

    count_iso = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count_iso += 1
                    break
    return count_iso / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    """Fraction of ``fake_graphs`` not isomorphic to any previously seen fake graph."""
    if len(fake_graphs) == 0:
        return 0.0

    count_non_unique = 0
    fake_evaluated = []

    for fake_g in fake_graphs:
        unique = True
        if fake_g.number_of_nodes() == 0:
            continue

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break

        if unique:
            fake_evaluated.append(fake_g)
    return (float(len(fake_graphs)) - count_non_unique) / float(len(fake_graphs))


def eval_VUN(fake_graphs, train_graphs, validity_func):
    """Fraction of Unique, Novel and Valid graphs from ``fake_graphs``."""
    count_non_unique = 0
    count_isomorphic = 0
    count_valid = 0
    fake_evaluated = []

    for fake_g in fake_graphs:
        unique = True
        if fake_g.number_of_nodes() == 0:
            continue

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break

        if not unique:
            continue

        fake_evaluated.append(fake_g)

        non_isomorphic_to_train = True
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count_isomorphic += 1
                    non_isomorphic_to_train = False
                    break

        if non_isomorphic_to_train:
            if validity_func(fake_g):
                count_valid += 1

    n_fake = float(len(fake_graphs))
    frac_unique = (n_fake - count_non_unique) / n_fake
    frac_UN = (n_fake - count_non_unique - count_isomorphic) / n_fake
    frac_VUN = count_valid / n_fake

    return frac_unique, frac_UN, frac_VUN


def main():
    data_name = 'sbm'
    method = 'hogdiff'

    gen_graphs_path = os.path.join(PROJECT_ROOT, 'analysis', 'samples', data_name, f'{method}_{data_name}_samples.pkl')
    with open(gen_graphs_path, 'rb') as f:
        gen_graphs = pickle.load(f)

    raw_path = os.path.join(PROJECT_ROOT, 'data', data_name, f'{data_name}.pkl')
    with open(raw_path, 'rb') as f:
        train_graphs, _, test_graphs = pickle.load(f)

    print(f"@@ test size:{len(test_graphs)}, sample size: {len(gen_graphs)}")

    uniqueness, UN, VUN = eval_VUN(gen_graphs, train_graphs, is_sbm_graph)
    print(f"frac_unique: {uniqueness}, frac_UN: {UN}, frac_VUN: {VUN}")


if __name__ == '__main__':
    main()
