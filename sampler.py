import time
import pickle

import os
from datetime import datetime
from evaluation.mol_nspdk_evaluator import nspdk_eval_fn
from utils.mol_utils import *

from evaluation.mol_evaluator import get_all_metrics

from utils import file_utils, solver, loader, graph_utils
from utils.path_manager import PathManager
from utils.dataloader import dataloader
from utils.file_utils import load_checkpoint
from utils.data_utils import data_masker, get_data_scaler
from utils.debug_utils import Timer
from evaluation.stats import eval_graph_list
from utils.dataloader import transform_adjs
from data import _GENERIC_DATASETS, _MOL_DATASETS

FCD_the = {'qm9': 2.0, 'zinc250k': 23.0}
VAL_the = {'qm9': 0.91, 'zinc250k': 0.91, 'moses': 0.80, 'guacamol': 0.70}
_DEBUG_ = os.getenv('_DEBUG_', 'false').lower() in ('1', 'true', 'yes')

class Sampler(object):
    def __init__(self, config, logger, mode='classical', device=None):

        self.mode = mode
        self.logger = logger
        self.data_name = config.data.name
        self.pl = file_utils.PathLoader(config, is_train=False)
        self.device = loader.load_device() if device is None else device
        self.config = config

        logger.log(f"{mode} Sampling with ckpt:{self.pl.get_ckpt_path()}")

        self.score_model, self.config = load_checkpoint(self.pl.get_ckpt_path(), self.device, config, self.mode)
        self.sampling_fn = solver.load_pc_sampler(self.config, self.device)

        self.latent_space = config.data.latent_space
        self.scaler = get_data_scaler(self.config)

        if mode == 'classical':
            self.train_ds = dataloader(config, mode='classical')
        elif mode == 'higher-order':
            self.train_ds = dataloader(config, mode='OU')
        else:
            self.train_ds = None


        if self.data_name in _MOL_DATASETS:
            self.num_sampling_rounds = int(np.ceil(self.config.eval.num_samples / self.config.eval.batch_size))
        elif self.data_name in _GENERIC_DATASETS:
            if self.data_name in ['sbm']:
                with open(self.pl.raw_graph_path, 'rb') as f:
                    _, _, self.test_graph_list = pickle.load(f)
            else:
                self.test_graph_list = file_utils.load_test_graphs(self.config, self.pl)
            self.num_samples = self.config.eval.batch_size
            self.num_sampling_rounds = int(np.ceil(self.num_samples / self.config.eval.batch_size))
            print(f"test size: {len(self.test_graph_list)}, number samples:{self.num_samples}, eval batch size:{self.config.eval.batch_size}")
        else:
            raise ValueError(f'{self.data_name} not supported yet!')


        self._log_configs()


    def _log_configs(self):
        """Logs the configuration settings."""
        self.logger.log('-' * 10 + f" Run exp [{self.config.exp_name}] at time: {self.logger.exp_time} with log: {self.logger.log_name}.log, seed={self.config.eval.seed} " + '-' * 10)

        config_categories = ['data', 'sde', 'model', 'optim', 'training', 'sampling', 'eval']
        for category in config_categories:
            self.logger.log_config(getattr(self.config, category), log_head=f"{category.capitalize()} log")

        self.logger.log('=' * 138)

    def sample(self, mu_list=None):
        all_atoms, all_bonds, all_sample_nodes, all_eig = [], [], [], []
        out_list = []
        start = time.perf_counter()
        loader.load_seed(self.config.eval.seed)
        for r in range(self.num_sampling_rounds):
            self.logger.log(f"sampling round: {r}")
            if mu_list is None:
                idx = torch.randint(0, len(self.train_ds), (self.config.eval.batch_size,))
                if self.mode == 'higher-order':
                    # train_ds = [x, la, u, adj, ho_x, ho_la, ho_u, ho_adj]
                    # ou_masker and u0 are used in phase-2 sampling.
                    ou_masker = data_masker(self.train_ds.tensors[3][idx], self.config, self.device)
                    u0 = self.train_ds.tensors[2][idx].to(self.device)
                    # ho_masker and ho_u are used in phase 1 (higher-order).
                    ho_masker = data_masker(self.train_ds.tensors[7][idx], self.config, self.device)
                    ho_u = self.train_ds.tensors[6][idx].to(self.device)
                    mu = {'u0': ho_u, 'masker': ho_masker}
                    ou_mu = {'ho_masker': ho_masker, 'u0': u0, 'masker': ou_masker}

                    if self.config.exp.plot:
                        plt_path = os.path.join(self.pl.plot_dir, "train.pth")
                        num_sample = self.config.eval.batch_size
                        if self.data_name in _GENERIC_DATASETS:
                            torch.save(self.train_ds.tensors[3][idx[:num_sample]], plt_path)
                        elif self.data_name in _MOL_DATASETS:
                            sample_nodes = ou_masker.get_atom_mask().sum(-1).to(torch.int)[:num_sample]
                            train_atom = self.train_ds.tensors[0][idx[:num_sample]]
                            train_bond = self.train_ds.tensors[3][idx[:num_sample]]
                            torch.save([train_atom, train_bond, sample_nodes], plt_path)

                elif self.mode == 'classical':
                    # train_ds = [x, la, u, adj]
                    masker = data_masker(self.train_ds.tensors[3][idx], self.config, self.device)
                    u0 = self.train_ds.tensors[2][idx].to(self.device)
                    mu = {'u0': u0, 'masker': masker}
            else:
                assert self.mode == 'OU', "Mode should be 'OU' when given mu."
                # Incoming mu is already scaled.
                mu = {key: (value.to('cpu').to(self.device) if value is not None else None) for key, value in mu_list[r].items()}



            x, la, adj = self.sampling_fn(self.score_model, mu)
            if self.mode == 'higher-order' and mu_list is None:
                if self.data_name in _MOL_DATASETS:
                    x, la, adj = self._mol_post_process(x, adj)
                    if self.config.exp.plot:
                        plt_path = os.path.join(self.pl.plot_dir, "ho.pth")
                        sample_nodes = ou_masker.get_atom_mask().sum(-1).to(torch.int)[:num_sample]
                        torch.save([x, adj, sample_nodes], plt_path)
                elif self.data_name in _GENERIC_DATASETS:
                    la, adj = self._generic_post_process(adj)
                # mu is fed directly to phase 2, so it needs scaling + masking here.
                ou_mu.update({'ho_x': x, 'ho_la': la,  'ho_adj': adj})
                out_mu = self._norm_mu(ou_mu)
                out_list.append(out_mu)

            elif self.mode in ['classical', 'OU']:
                all_bonds.append(adj)
                if _DEBUG_:
                    all_eig.append(la)
                if self.data_name in _MOL_DATASETS:
                    all_atoms.append(x)
                    all_sample_nodes.append(mu['masker'].get_atom_mask().sum(-1).to(torch.int))

        end = time.perf_counter()
        print(f"Sampling time: {end - start:.6f} s")

        result_dict = {}
        if self.mode in ['classical', 'OU']:
            if self.data_name in _MOL_DATASETS:
                result_dict = self._mol_eval_fn(all_atoms, all_bonds, all_sample_nodes)
            elif self.data_name in _GENERIC_DATASETS:
                result_dict = self._generic_eval_fn(all_bonds)

            self.logger.log('-' * 30 + 'Evaluaction results' + '-' * 30)
            for metric, value in result_dict.items():
                self.logger.log(f'{metric}: {value}')

        self.logger.log('-'*30 + f"Finish {self.mode} sampling" + '-'*30)
        self.logger.log('=' * 138)

        return out_list, result_dict

    def _norm_mu(self, mu):
        # The computed x is inverse-scaled; rescale before feeding it back as mu.
        if self.data_name in _MOL_DATASETS:
            masker, ho_masker = mu['masker'], mu['ho_masker']
            ho_x = ho_masker.mask_atom(self.scaler(mu['ho_x'], type='atom'))
            mu['ho_x'] = ho_x + masker.mask_atom(torch.randn_like(ho_x, device='cpu').to(ho_x.device))
            mu['ho_la'] = ho_masker.mask_la(self.scaler(mu['ho_la'], type='eig'))
            mu['ho_adj'] = ho_masker.mask_bond(self.scaler(mu['ho_adj'], type='bond'))

        return mu

    def _mol_post_process(self, atom_sample, bond_sample):
        max_indices = torch.argmax(atom_sample, dim=2, keepdim=True)
        x = torch.zeros_like(atom_sample).scatter_(2, max_indices, 1)
        adjs = bond_sample * 3
        adjs[adjs >= 2.5] = 3
        adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
        adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
        adjs[adjs < 0.5] = 0
        adjs = adjs / 3.0
        la, u = transform_adjs(adjs, self.latent_space)
        return x, la, adjs

    def _generic_post_process(self, adjs):
        adjs = torch.where(adjs < 0.5, torch.zeros_like(adjs), torch.ones_like(adjs))
        la, u = transform_adjs(adjs, self.latent_space)
        return la, adjs

    def _mol_eval_fn(self, all_atoms, all_bonds, all_sample_nodes):
        num_samples = self.config.eval.num_samples
        atom_sample = torch.concat(all_atoms, dim=0)[:num_samples]
        bond_sample = torch.concat(all_bonds, dim=0)[:num_samples]
        sample_nodes = torch.concat(all_sample_nodes, dim=0)[:num_samples]

        all_samples, all_valid_wd = tensor2mol(atom_sample, bond_sample, sample_nodes, self.data_name)

        gen_smile_list = [Chem.MolToSmiles(mol) for mol in all_samples if mol is not None]

        if getattr(self.config.eval, 'save_samples', False) or len(gen_smile_list) > 9000:
            sample_rel_path = os.path.join(PathManager.PRO_ROOT, 'analysis', 'samples', f"{self.data_name}", f"hogdiff_{self.data_name}_smiles({datetime.now().strftime('%Y%m%d_%H%M%S')}).txt")
            os.makedirs(os.path.dirname(sample_rel_path), exist_ok=True)
            with open(sample_rel_path, "w") as f:
                for smi in gen_smile_list:
                    f.write(smi + "\n")
            print(f"Saved {len(gen_smile_list)} SMILES to {os.path.abspath(sample_rel_path)}")

        validity = np.sum(all_valid_wd) / len(all_valid_wd)
        result_dict = {'validity': validity}
        self.logger.log('Number of molecules: %d' % len(all_samples))
        self.logger.log(f"validity w/o corr.: {validity:.6f}")
        if validity >= VAL_the[self.data_name]:
            train_smiles, test_smiles = file_utils.load_smiles(self.data_name)
            with Timer(name="Evaluating metrics", enabled=True):
                metrics = get_all_metrics(gen=gen_smile_list, test=test_smiles, train=train_smiles, device=self.device, n_jobs=8)
                result_dict.update(metrics)


            if 'NSPDK' in self.config.eval.metrics and result_dict['FCD'] <= FCD_the[self.data_name]:
                result_dict['NSPDK'] = nspdk_eval_fn(all_samples, test_smiles)
            return result_dict



    def _generic_eval_fn(self, all_bonds):
        adjs = torch.concat(all_bonds, dim=0)[:self.num_samples]
        gen_graph_list = graph_utils.tensor2nxgraphs(adjs)

        if getattr(self.config.eval, 'save_samples', False):
            gen_graphs_save_path = os.path.join(PathManager.PRO_ROOT, 'analysis', 'samples', f"{self.data_name}/hogdiff_{self.data_name}_samples({datetime.now().strftime('%Y%m%d_%H%M%S')}).pkl")
            os.makedirs(os.path.dirname(gen_graphs_save_path), exist_ok=True)
            with open(gen_graphs_save_path, "wb") as f:
                pickle.dump(gen_graph_list, f)
                print(f"Saved {len(gen_graph_list)} generated graphs to {os.path.abspath(gen_graphs_save_path)}")

        if self.config.exp.plot:
            adj_path = os.path.join(self.pl.plot_dir, f"{self.mode}.pth")
            torch.save(adjs, adj_path)

        print(f"@@ test size:{len(self.test_graph_list)}, sample size: {len(gen_graph_list)}")
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, self.pl)
        result_dict['OU-ave'] = sum(result_dict.values()) / len(result_dict)


        return result_dict


def get_mu_from_ho_directedly(train_ds, config, device=None):
    """Use higher-order information directly as the phase-2 input."""
    rng = torch.Generator()
    rng.manual_seed(config.eval.seed)
    mu_list = []
    num_sampling_rounds = int(np.ceil(config.eval.num_samples / config.eval.batch_size)) if config.data.name in _MOL_DATASETS else 1
    for r in range(num_sampling_rounds):

        idx = torch.randint(0, len(train_ds), (config.eval.batch_size,), generator=rng)
        scaler = get_data_scaler(config)
        # train_ds = [x, la, u, adj, ho_x, ho_la, ho_u, ho_adj]
        u0 = train_ds.tensors[2][idx]
        masker = data_masker(train_ds.tensors[3][idx], config)
        ho_masker = data_masker(train_ds.tensors[7][idx], config)

        ho_x = ho_masker.mask_atom(scaler(train_ds.tensors[4][idx], type='atom'))
        ho_la = ho_masker.mask_la(scaler(train_ds.tensors[5][idx], type='eig'))
        ho_adj = ho_masker.mask_bond(scaler(train_ds.tensors[7][idx], type='bond'))
        if config.data.name in _MOL_DATASETS:
            ho_x = ho_x + masker.mask_atom(torch.randn_like(ho_x))


        mu = {'ho_x': ho_x, 'ho_la': ho_la, 'ho_adj': ho_adj, 'u0': u0, 'ho_masker': ho_masker, 'masker': masker}
        if _DEBUG_:
            mu['la'] = train_ds.tensors[1][idx]
            mu['adj'] = train_ds.tensors[3][idx]


        if device is not None:
            mu = {key: (value.to(device) if value is not None else None) for key, value in mu.items()}
        mu_list.append(mu)

    return mu_list
