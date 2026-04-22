import torch
import os
import pickle
import logging
import pandas as pd
import json
import numpy as np
from models.ema import ExponentialMovingAverage
from utils import loader
from data import _GENERIC_DATASETS, _MOL_DATASETS
from utils.path_manager import PathManager


def get_root_path():
    """ Always return the root path. """
    from pathlib import Path
    import inspect
    curr_file = Path(inspect.getfile(inspect.currentframe())).parent.parent
    return curr_file

def get_config_path(config_name):
    if not config_name.endswith('.yaml'):
        config_name = config_name + '.yaml'
    return os.path.join(get_root_path(), 'configs', config_name)

"""
NOTE: PathLoader is being gradually replaced by utils.path_manager.PathManager.
"""
class PathLoader(object):
    root_path = get_root_path()
    data_dir = os.getenv('DATA_ROOT', str(os.path.join(root_path, 'data')))
    def __init__(self, config, is_train=None, exp_time=None):
        self.data_name = config.data.name.lower()
        self.exp_name = exp_name = config.exp_name
        self.CKPT_ROOT = os.getenv('CKPT_ROOT', str(PathLoader.root_path))
        self.LOG_ROOT = os.getenv('LOG_ROOT', str(PathLoader.root_path))
        LOG_FOLDER = os.getenv('LOG_FOLDER', str(exp_name))


        self.log_dir = os.path.join(self.LOG_ROOT, 'logs', self.data_name, LOG_FOLDER)
        self.ckpt_dir = os.path.join(self.CKPT_ROOT, "checkpoints", self.data_name, exp_name)
        self.plot_dir = os.path.join(self.LOG_ROOT, 'analysis', 'figs', self.data_name, exp_name)

        if is_train:
            meta = getattr(config, 'ckpt_meta', None)
            if meta:
                if not os.path.isabs(meta):
                    meta = os.path.join(self.CKPT_ROOT, meta)
                self.ckpt_name = os.path.splitext(os.path.basename(meta))[0]
                self.meta_ckpt_path = meta
            else:
                self.ckpt_name = exp_time
                self.meta_ckpt_path = None
        elif is_train is False:
            ckpt = config.ckpt
            if not os.path.isabs(ckpt):
                ckpt = os.path.join(self.CKPT_ROOT, ckpt)
            self.ckpt_name = self.ckpt_path = ckpt
            if os.path.isfile(self.ckpt_name):
                file_name = os.path.basename(self.ckpt_name)
                # Strip the ".pth" extension.
                self.ckpt_name = os.path.splitext(file_name)[0]
            assert os.path.exists(self.get_ckpt_path()), f"Not found checkpoint {self.get_ckpt_path()}"


        if self.data_name in _MOL_DATASETS:
            # qm9 total:133885, train:120803, test:13082
            self.raw_mol_path = os.path.join(PathLoader.data_dir, self.data_name, 'raw', f'{self.data_name}_property.csv')
            self.valid_idx_path = os.path.join(PathLoader.data_dir, self.data_name, 'raw', f'valid_idx_{self.data_name}.json')
            self.atom_num_list_path = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'atom_num_list.npy')
            self.ref_nspdk_stats_path = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'ref_nspdk_stats.npy')
            self.processed_data_dir = os.path.join(PathLoader.data_dir, self.data_name, 'processed')
        elif self.data_name in _GENERIC_DATASETS:
            self.raw_graph_path = os.path.join(PathLoader.data_dir, self.data_name,  f'{self.data_name}.pkl')
            self.cached_d1_degree = os.path.join(PathLoader.data_dir, self.data_name,'processed', 'cached_d1_degree.npy')
            self.cached_d1_orbit = os.path.join(PathLoader.data_dir, self.data_name,'processed', 'cached_d1_orbit.npy')
            self.cached_d1_cluster = os.path.join(PathLoader.data_dir, self.data_name,'processed', 'cached_d1_cluster.npy')
            self.cached_d1_spectral = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'cached_d1_spectral.npy')
            self.cached_hist_degree = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'cached_hist_degree.npz')
            self.cached_hist_orbit = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'cached_hist_orbit.npz')
            self.cached_hist_cluster = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'cached_hist_cluster.npz')
            self.cached_hist_spectral = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'cached_hist_spectral.npz')

        self.processed_path = os.path.join(PathLoader.data_dir, self.data_name, 'processed', 'atom_bond.pt')
        dir_to_create = [self.ckpt_dir,
                         self.log_dir,
                         os.path.dirname(self.processed_path),
                         self.plot_dir]

        self._create_dirs_(dir_to_create)






    def get_ho_file_path(self, lifting_type, min_media_size=None, max_media_size=None):
        """
        @ ho_file_path:
        - Tensor for mols,
        - List[nx.Graphs] for general graphs
        """
        if lifting_type == 'SCs':
            assert min_media_size is not None, 'min_media_size must be provided for lifting_type SCs!'
            tag = min_media_size
        elif lifting_type == 'CCs':
            assert max_media_size is not None, 'max_media_size must be provided for lifting_type CCs!'
            tag = max_media_size

        ho_file_path = os.path.join(PathLoader.data_dir, self.data_name, 'processed', f'{self.data_name}-lifting-{lifting_type}({tag}).pt')

        return ho_file_path



    def get_ckpt_path(self, step=None):
        if hasattr(self, 'ckpt_path') and os.path.isfile(self.ckpt_path):
            return self.ckpt_path

        if step is not None:
            ckpt_name = f"{self.ckpt_name}_{step}.pth"
        else:
            ckpt_name = self.ckpt_name + '.pth'


        return os.path.join(self.ckpt_dir, ckpt_name)

    def get_out_mu_path(self, step=None):
        if step is not None:
            ckpt_name = f"{self.ckpt_name}_{step}.pth"
        else:
            ckpt_name = self.ckpt_name + '.pth'


        return os.path.join(self.mu_dir, ckpt_name)

    def _create_dirs_(self, dir_list):
        for dir in dir_list:
            os.makedirs(dir, exist_ok=True)





def save_checkpoint(ckpt_path, state, config, device, mode='classical'):
    ckpt_dict = torch.load(ckpt_path, map_location=device) if os.path.exists(ckpt_path) else {}
    ckpt_dict[mode] = {
        'config': config,
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step'],
    }
    torch.save(ckpt_dict, ckpt_path)

def restore_checkpoint(config, ckpt_path, device, mode='classical'):
    """Trainer entry: try to restore from ``ckpt_meta``; fall back to a fresh state."""
    restore_flag = False
    if ckpt_path is None or not os.path.exists(ckpt_path):
        if ckpt_path is not None:
            logging.warning(f"No checkpoint found at {ckpt_path}. Returned the same state as input")
    else:
        loaded_state = torch.load(ckpt_path, map_location=device)
        if mode not in loaded_state:
            logging.warning(f"{mode} in checkpoint {ckpt_path} is not supported. Returned the same state as input")
        else:
            logging.info(f"Restoring checkpoint {ckpt_path}")
            loaded_state = loaded_state[mode]
            config = loaded_state['config']
            restore_flag = True

    score_model = loader.load_score_model(config, device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = loader.load_optimizer(config, score_model.parameters())
    step = 0


    if restore_flag:
        score_model.load_state_dict(loaded_state['model'], strict=True)
        ema.load_state_dict(loaded_state['ema'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        step = loaded_state['step']

    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=step)

    return state, config


#from utils.loader import seed_context
def load_checkpoint(ckpt_path, device, sample_config=None, mode='classical'):
    """Sampler entry: load the checkpoint at ``ckpt_path``."""
    assert os.path.exists(ckpt_path), f"No checkpoint found at {ckpt_path}"
    loaded_state = torch.load(ckpt_path, map_location=device)
    assert mode in loaded_state, f"Checkpoint {ckpt_path} has no mode {mode}"

    loaded_state = loaded_state[mode]
    configt = loaded_state['config']

    #with seed_context(999):
    # Initialize model
    score_model = loader.load_score_model(configt, device)
    #optimizer = loader.load_optimizer(sample_config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=configt.model.ema_rate)

    # load
    score_model.load_state_dict(loaded_state['model'], strict=True)
    ema.load_state_dict(loaded_state['ema'])
    #optimizer.load_state_dict(loaded_state['optimizer'])

    ema.copy_to(score_model.parameters())

    #state = dict(config=configt, optimizer=optimizer, model=score_model, ema=ema, step=step)
    if sample_config is not None:
        configt.eval = sample_config.eval
        configt.sampling = sample_config.sampling
        configt.ckpt = sample_config.ckpt
        configt.exp = sample_config.exp

    return score_model, configt


def load_smiles(dataset='qm9'):
    dataset=dataset.lower()
    data_raw_dir = PathManager.DATA_RAW_DIR

    if dataset in ['qm9', 'zinc250k']:
        col = 'smile'
        df = pd.read_csv(os.path.join(data_raw_dir, f'{dataset}_property.csv') )

        with open(os.path.join(data_raw_dir, f'valid_idx_{dataset}.json')  ) as f:
            test_idx = json.load(f)

        if dataset in ['qm9']:
            test_idx = list(map(int, test_idx['valid_idxs']))

        test_smiles = list(df[col].loc[test_idx])

        train_idx = list(set(np.arange(len(df))).difference(set(test_idx)))
        train_smiles = list(df[col].loc[train_idx])

        # eval_idx = train_idx[torch.randperm(len(train_idx))[:len(test_idx)]]
        # eval_smiles = list(df[col].loc[eval_idx])



        # train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
    elif dataset in ['guacamol']:
        with open(os.path.join(data_raw_dir, f'new_train.smiles'), 'r') as f:
            train_smiles = [line.strip() for line in f if line.strip()]
        with open(os.path.join(data_raw_dir, f'new_test.smiles'), 'r') as f:
            test_smiles = [line.strip() for line in f if line.strip()]
    elif dataset in ['moses']:
        train_smiles = list(pd.read_csv(os.path.join(data_raw_dir, f'train_moses.csv'), sep=',', dtype='str')['SMILES'])
        test_smiles = list(pd.read_csv(os.path.join(data_raw_dir, f'test_moses_scaffolds.csv'), sep=',', dtype='str')['SMILES'])
    else:
        raise ValueError(f'{dataset} not supported.')

    return train_smiles, test_smiles


def load_test_graphs(config, pathloader):

    with open(pathloader.raw_graph_path, 'rb') as f:
        pairwise_graph_list = pickle.load(f)
    test_size = int(config.data.test_split * len(pairwise_graph_list))
    test_pairwise_graph_list = pairwise_graph_list[:test_size]
    return test_pairwise_graph_list



