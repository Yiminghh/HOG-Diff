import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.debug_utils import Timer
from utils import file_utils
from utils.config_paraser import load_config
from utils.graph_utils import tensor2nxgraphs
from utils.dataloader import dataloader


def compute_average_node_degree(graph_list):
    """
    计算图列表中所有节点的平均度数

    参数:
        graph_list: List[nx.Graph]，每个元素是一个 NetworkX 图

    返回:
        avg_degree: 所有节点的平均度数（float）
    """
    total_degree = 0
    total_nodes = 0

    for g in graph_list:
        degrees = dict(g.degree())
        total_degree += sum(degrees.values())
        total_nodes += len(degrees)

    avg_degree = total_degree / total_nodes if total_nodes > 0 else 0
    return avg_degree


def preprocess(data_dir='data', dataset='sbm', measure_train_mmd=False):
    config = load_config(dataset + '.yaml', 'test', mode='OU')
    pl = file_utils.PathLoader(config)
    with open(pl.raw_graph_path, 'rb') as f:
        train_graphs, _, test_graphs = pickle.load(f)

    train_ds = dataloader(config, mode='OU')
    train_graphs = tensor2nxgraphs(train_ds.tensors[3])

    with Timer(f"len(train):{len(train_graphs)},len(test):{len(test_graphs)}, 计算mmd用时"):
        if measure_train_mmd:
            from evaluation.stats import degree_stats, orbit_stats_all, clustering_stats, spectral_stats
            from evaluation.mmd import gaussian_tv

            kernel = gaussian_tv
            train_mmd_degree = degree_stats(test_graphs, train_graphs, kernel, pathloader=pl)
            train_mmd_4orbits = orbit_stats_all(test_graphs, train_graphs, kernel, pathloader=pl)
            train_mmd_clustering = clustering_stats(test_graphs, train_graphs, kernel, pathloader=pl)
            train_mmd_spectral = spectral_stats(test_graphs, train_graphs, kernel, pathloader=pl)
            print('TV measures of Training set vs Test set: ')
            print(f'Deg.: {train_mmd_degree:.4f}, Clus.: {train_mmd_clustering:.4f} '
                  f'Orbits: {train_mmd_4orbits:.4f}, Spec.: {train_mmd_spectral:.4f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sbm')
    parser.add_argument('--mmd', default=True)
    args = parser.parse_known_args()[0]

    preprocess(dataset=args.dataset, measure_train_mmd=args.mmd)
