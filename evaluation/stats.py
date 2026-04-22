import concurrent.futures
import logging
import os
import subprocess as sp
from datetime import datetime
import random
import sys
from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import os
from utils.file_utils import get_root_path
from evaluation.mmd import process_tensor, compute_mmd, gaussian, gaussian_emd, compute_nspdk_mmd, gaussian_tv
import time

PRINT_TIME = False 
# -------- the relative path to the orca dir --------
# ORCA_DIR = 'evaluation/orca'
ORCA_DIR = os.path.join(get_root_path(), 'evaluation', 'orca')

def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y

# -------- Compute degree MMD --------
def degree_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True, pathloader=None):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # -------- in case an empty graph is generated --------
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if pathloader is not None and os.path.exists(pathloader.cached_hist_degree):
        loaded = np.load(pathloader.cached_hist_degree)
        sample_ref = [loaded[key] for key in loaded.files]
    else:
        if is_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for deg_hist in executor.map(degree_worker, graph_ref_list):
                    sample_ref.append(deg_hist)
        else:
            for i in range(len(graph_ref_list)):
                degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
                sample_ref.append(degree_temp)
        np.savez(pathloader.cached_hist_degree, *sample_ref)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, cached_d1_path=pathloader.cached_d1_degree)
    return mmd_dist


def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


# -------- Compute spectral MMD --------
def spectral_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True, pathloader=None):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    if pathloader is not None and os.path.exists(pathloader.cached_hist_spectral):
        loaded = np.load(pathloader.cached_hist_spectral)
        sample_ref = [loaded[key] for key in loaded.files]
    else:
        if is_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for spectral_density in executor.map(spectral_worker, graph_ref_list):
                    sample_ref.append(spectral_density)
        else:
            for i in range(len(graph_ref_list)):
                spectral_temp = spectral_worker(graph_ref_list[i])
                sample_ref.append(spectral_temp)
        np.savez(pathloader.cached_hist_spectral, *sample_ref)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, cached_d1_path=pathloader.cached_d1_spectral)

    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    # print("clustering_coeffs_list:",clustering_coeffs_list)
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


# -------- Compute clustering coefficients MMD --------
# def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, is_parallel=True):
def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, is_parallel=True, pathloader=None):
    print("bins number:", bins)
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    # print("graph_ref_list:",graph_ref_list)

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if pathloader is not None and os.path.exists(pathloader.cached_hist_cluster):
        loaded = np.load(pathloader.cached_hist_cluster)
        sample_ref = [loaded[key] for key in loaded.files]
    else:
        if is_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for clustering_hist in executor.map(clustering_worker,
                                                    [(G, bins) for G in graph_ref_list]):
                    sample_ref.append(clustering_hist)
        else:
            for i in range(len(graph_ref_list)):
                clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
                print("clustering_coeffs_list:", clustering_coeffs_list)
                hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
                sample_ref.append(hist)
        np.savez(pathloader.cached_hist_cluster, *sample_ref)


    try:
        # print("sample_ref:",sample_ref)
        # print("sample_pred:", sample_pred)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, 
                            sigma=1.0 / 10, distance_scaling=bins, cached_d1_path=pathloader.cached_d1_cluster)
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, sigma=1.0 / 10, cached_d1_path=pathloader.cached_d1_cluster)

    return mmd_dist


# -------- maps motif/orbit name string to its corresponding list of indices from orca output --------
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}

# ORCA uses a platform-specific line terminator after this header.
if sys.platform.startswith('win'):
    COUNT_START_STR = 'orbit counts: \r\n'
else:
    # linux, darwin, and other POSIX platforms use LF.
    COUNT_START_STR = 'orbit counts: \n'

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_file_path = os.path.join(ORCA_DIR, f'tmp-{random.random():.4f}.txt')
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    # print("output:", output)
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, KERNEL=gaussian, pathloader=None):
    # print("in orbit_stats_all")
    total_counts_pred = []

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            print('orca failed')
            continue
        # print("Predict  number_of_nodes:", G.number_of_nodes(), np.sum(orbit_counts, axis=0))
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    if pathloader is not None and os.path.exists(pathloader.cached_hist_orbit):
        loaded = np.load(pathloader.cached_hist_orbit)
        total_counts_ref = [loaded[key] for key in loaded.files]
    else:
        total_counts_ref = []
        for G in graph_ref_list:
            try:
                # print("try to count orca(G)")
                orbit_counts = orca(G)
            except Exception as e:
                print(e)
                continue
            # print("Label  number_of_nodes:", G.number_of_nodes(), np.sum(orbit_counts, axis=0))
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
            total_counts_ref.append(orbit_counts_graph)
        np.savez(pathloader.cached_hist_orbit, *total_counts_ref)


    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=KERNEL,
                           is_hist=False, sigma=30.0, cached_d1_path=pathloader.cached_d1_orbit)


    return mmd_dist

##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats
}


# -------- Evaluate generated generic graphs --------
def eval_graph_list(graph_ref_list, graph_pred_list, pathloader=None, methods=None):
    if methods is None:
        methods = ['degree', 'cluster', 'orbit']

    kernels = {'degree': gaussian_emd,
               'cluster': gaussian_emd,
               'orbit': gaussian}

    if pathloader.data_name in ['sbm']:
        print(f"SBM uses TV distance kernel.")
        methods = ['degree', 'cluster', 'orbit','spectral']
        kernels = {'degree': gaussian_tv,
                'cluster': gaussian_tv,
                'orbit': gaussian_tv,
                'spectral': gaussian_tv}

    results = {}
    for method in methods:
        time1 = time.time()
        if method == 'nspdk':
            results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
        else:
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, kernels[method], pathloader=pathloader), 6)
        logging.info(f"{method}: {results[method]}, Time {time.time() - time1}")

    return results
