import os
from torch_geometric.loader import DataLoader
import pickle
import sampler
from utils import solver
from utils.file_utils import load_checkpoint
from utils.mol_utils import *
from utils.file_utils import restore_checkpoint, save_checkpoint
import random
from utils.data_utils import data_masker, get_data_scaler
import itertools
from utils.dataloader import transform_adjs
from utils.dataloader import dataloader, load_batch
from utils import losses, file_utils, loader, graph_utils
from utils.loader import seed_context
from evaluation.stats import eval_graph_list
from data import _GENERIC_DATASETS, _MOL_DATASETS
from evaluation.mol_evaluator import get_all_metrics
from utils.debug_utils import CheckpointManager
import wandb

STOP_THR = {'qm9':19, 'zinc250k':19, 'ego_small':15, 'community_small':15, 'enzymes':8, 'sbm':12,
            'guacamol':10, 'moses':10}


class Trainer(object):
    def __init__(self, config, logger, mode='classical'):

        """Runs the training pipeline of HOG-Diff."""
        self.data_name = config.data.name.lower()
        self.config = config
        self.ct = config.training
        self.mode = mode
        self.logger = logger
        self.device = loader.load_device()
        self.pl = file_utils.PathLoader(config, is_train=True, exp_time=logger.exp_time)

        # Optional: when `fix_ckpt` is set in the config, phase-1 (higher-order)
        # training can be evaluated against a pre-trained phase-2 (OU) checkpoint.
        # Provide the path via `config.eval.fix_ckpt_path` (str or list).
        if self.mode == 'higher-order' and getattr(config.eval, 'fix_ckpt', False):
            self.ou_ckpt_path = getattr(config.eval, 'fix_ckpt_path', None)
            if isinstance(self.ou_ckpt_path, list):
                self.ou_ckpt_path = random.choice(self.ou_ckpt_path)
        else:
            self.ou_ckpt_path = None


        if self.ct.snapshot_sampling:
            # Force corrector off during training — otherwise per-step sampling is too costly.
            self.config.sampling.corrector = 'None'
            if self.data_name == 'qm9':
                self.config.eval.batch_size = self.config.eval.num_samples = 512
            elif self.data_name in ['zinc250k', 'guacamol', 'moses']:
                self.config.eval.batch_size = self.config.eval.num_samples = 256

        # Resume training when --ckpt_meta is provided; otherwise initialize from scratch
        self.state, self.config = restore_checkpoint(self.config, self.pl.meta_ckpt_path, self.device, mode)
        if self.pl.meta_ckpt_path is not None:
            wandb.summary['ckpt_path'] = self.pl.meta_ckpt_path
        self.score_model, self.ema, self.initial_step = self.state['model'], self.state['ema'], int(self.state['step'])

        self.train_ds = dataloader(self.config, mode)
        num_workers = getattr(config.data, 'num_workers', 16)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.ct.batch_size, shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True if self.device.type == 'cuda' else False,
                                        persistent_workers=True if num_workers > 0 else False,)

        if self.ct.snapshot_sampling:
            ou_train_ds = self.train_ds if mode == 'OU' else dataloader(config, mode='OU')
            if self.mode == 'OU':
                self.ou_mu = sampler.get_mu_from_ho_directedly(ou_train_ds, self.config, self.device)[0]
            elif self.mode == 'higher-order' and self.ou_ckpt_path is not None:
                with seed_context(self.config.eval.seed):
                    self.ou_score_model, self.ou_config = load_checkpoint(self.ou_ckpt_path, self.device, mode='OU')
                    self.ou_sampling_fn = solver.load_pc_sampler(self.ou_config, self.device)
                    self.mu, self.ou_mu = gen_mu_for_ho_sampling(ou_train_ds, self.ou_config, self.device)
                    self.scaler = get_data_scaler(self.config)
                    self.latent_space = config.data.latent_space
                    self.logger.log(f"@@ Ho training with fixed OU ckpt: {self.ou_ckpt_path}")
                    wandb.summary['OU_ckpt'] = self.ou_ckpt_path

            elif self.mode == 'higher-order' and self.ou_ckpt_path is None:
                self.ct.snapshot_sampling = False

            self.sampling_fn = solver.load_pc_sampler(self.config, self.device)
            self._load_sampling_ref()
            if self.data_name == 'moses':
                self.ckpt_manager = CheckpointManager({'validity': 0.85, 'FCD': 8.0, 'Scaf': 0.01, 'SNN': 0.35})
            elif self.data_name == 'guacamol':
                self.ckpt_manager = CheckpointManager({'validity': 0.75, 'FCD': 15.0}, max_size=3)
            else:
                self.ckpt_manager = CheckpointManager({'validity': 0.90, 'FCD': 4.0, 'NSPDK': 0.01})




        self._log_configs()
        if self.ou_ckpt_path is not None:
            self._log_configs(self.ou_config)




    def _log_configs(self, config=None):
        cf = self.config if config is None else config
        """Logs the configuration settings."""
        self.logger.log('-' * 10 + f" Run exp [{cf.exp_name}] at time: {self.logger.exp_time} with log: {self.logger.log_name}.log, seed={cf.training.seed} " + '-' * 10)
        self.logger.log(f"Starting {self.mode} training loop at step {self.initial_step}.")

        config_categories = ['data', 'sde', 'model', 'optim', 'training', 'sampling', 'eval']
        for category in config_categories:
            self.logger.log_config(getattr(cf, category), log_head=f"{category.capitalize()} log")

        self.logger.log('=' * 138)

    def _load_sampling_ref(self):
        with torch.no_grad():
            if self.data_name in ['sbm']:
                with open(self.pl.raw_graph_path, 'rb') as f:
                    _, _, self.test_graph_list = pickle.load(f)

            elif self.data_name in _MOL_DATASETS:
                _, self.test_smiles = file_utils.load_smiles(self.data_name)
            elif self.data_name in _GENERIC_DATASETS:
                self.test_graph_list = file_utils.load_test_graphs(self.config, self.pl)
            else:
                raise ValueError(f'{self.data_name} not supported.')



    def train(self):
        train_step_fn = losses.get_step_fn(self.config, train=True)

        min_loss, self.early_stop_count = 9999, 0
        min_error = 9999  # for generic datasets
        self.max_validity, self.max_SNN, self.max_Scaf, self.min_FCD  = 0, 0, 0, 9999  # for mol datasets
        train_iter = itertools.cycle(self.train_loader)

        for step in range(self.initial_step, self.ct.n_iters + 1):
            graphs = next(train_iter)
            sample_con = False
            if self.early_stop_count >= STOP_THR[self.data_name]:
                break

            batch = load_batch(graphs, self.config, self.device, self.mode)

            loss, loss_atom, loss_bond = train_step_fn(self.state, batch)

            if step % self.ct.log_freq == 0:
                self.logger.log(f"step: {step}, training_loss: {loss.item():.5e} (loss_atom:{loss_atom.item():.5f}, loss_bond:{loss_bond.item():.5f})")
                wandb.log({'train/loss': float(loss.item()),
                            'train/loss_atom': float(loss_atom.item()),
                            'train/loss_bond': float(loss_bond.item()),
                        }, step=step)
                if loss.item() < min_loss and step >= 10000:
                    min_loss = loss.item()
                    sample_con = True

            if self.ct.snapshot_sampling and (step % self.ct.snapshot_freq == 0 or step == self.ct.n_iters or sample_con) and step != 0:
                if self.data_name in _GENERIC_DATASETS:
                    save_checkpoint(self.pl.get_ckpt_path(step), self.state, self.config, self.device, self.mode)

                if self.ct.snapshot_sampling:
                    with torch.no_grad(), seed_context(self.config.eval.seed):
                        self.ema.store(self.score_model.parameters())
                        self.ema.copy_to(self.score_model.parameters())
                        if self.data_name in _MOL_DATASETS and self.mode in ['OU', 'higher-order']:
                            gen_metrics = self._mol_eval_fn(step, sample_con)
                        elif self.data_name in _GENERIC_DATASETS and self.mode in ['OU', 'higher-order']:
                            gen_metrics = self._generic_eval_fn(step, min_error, sample_con)
                        else:
                            raise NotImplementedError
                        self.ema.restore(self.score_model.parameters())

                        if gen_metrics is not None:
                            ckpt_path = self.pl.get_ckpt_path(step)
                            should_save, to_remove = self.ckpt_manager.update(ckpt_path, gen_metrics)
                            for rem in to_remove:
                                try:
                                    os.remove(rem)
                                    self.logger.log(f"Remove checkpoint: {rem}")
                                except:
                                    pass
                            if should_save:
                                save_checkpoint(ckpt_path, self.state, self.config, self.device, self.mode)
                                self.logger.log(f"Save checkpoint: {ckpt_path}")


        self.logger.log('-' * 30 + f"Finish {self.mode} training" + '-' * 30)
        self.logger.log('=' * 138)

        if self.data_name in ['moses']:
            wandb.run.summary['min_train_loss'] = float(min_loss)
            wandb.run.summary['min_FCD'] = float(self.min_FCD)
            wandb.run.summary['max_validity'] = float(self.max_validity)
            wandb.run.summary['max_SNN'] = float(self.max_SNN)
            wandb.run.summary['max_Scaf'] = float(self.max_Scaf)
            wandb.run.name = f"{self.data_name}_FCD{self.min_FCD:.4f}_SNN{self.max_SNN:.4f}_Scaf{self.max_Scaf:.4f}"
        elif self.data_name in ['guacamol']:
            wandb.run.summary['min_train_loss'] = float(min_loss)
            wandb.run.summary['min_FCD'] = float(self.min_FCD)
            wandb.run.name = f"{self.data_name}_FCD{self.min_FCD:.4f}"
        elif self.data_name in _GENERIC_DATASETS:
            wandb.run.summary['min_train_loss'] = float(min_loss)
            wandb.run.summary['min_ave_error'] = float(min_error)
            wandb.run.name = f"{self.data_name}_Avg{min_error:.4f}"




    def _mol_eval_fn(self, step,  sample_con):
        metrics = None
        if self.mode == 'higher-order' and self.ou_ckpt_path is None:
            return metrics
        if not sample_con:
            self.early_stop_count += 1

        try:
            if self.mode == 'OU':
                atom_sample, _, bond_sample = self.sampling_fn(self.score_model, self.ou_mu)
            elif self.mode == 'higher-order':
                ho_x, ho_la, ho_adj = self.sampling_fn(self.score_model, self.mu)
                def _mol_post_process(atom_sample, bond_sample, latent_space):
                    max_indices = torch.argmax(atom_sample, dim=2, keepdim=True)
                    x = torch.zeros_like(atom_sample).scatter_(2, max_indices, 1)
                    adjs = bond_sample * 3
                    adjs[adjs >= 2.5] = 3
                    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
                    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
                    adjs[adjs < 0.5] = 0
                    adjs = adjs / 3.0
                    la, u = transform_adjs(adjs, latent_space)
                    return x, la, adjs

                ho_x, ho_la, ho_adj = _mol_post_process(ho_x, ho_adj, self.latent_space)
                ho_masker = self.ou_mu['ho_masker']
                ho_x = ho_masker.mask_atom(self.scaler(ho_x, type='atom'))
                ou_mu = {'ho_x': ho_x, 'ho_la': ho_la,  'ho_adj': ho_adj}
                ou_mu.update(self.ou_mu)

                atom_sample, _, bond_sample = self.ou_sampling_fn(self.ou_score_model, ou_mu)


            sample_nodes = self.ou_mu['masker'].get_atom_mask().sum(-1).to(torch.int)

            sample_list, valid_wd = tensor2mol(atom_sample, bond_sample, sample_nodes, self.config.data.name)
            valid_wd_rate = np.sum(valid_wd) / len(valid_wd)
            metrics = {'validity': valid_wd_rate}
            self.logger.log('-' * 30 + f'step={step}(early_stop_count={self.early_stop_count}), Evaluaction results' + '-' * 30)
            self.logger.log(f"step: {step}, n_mol: {len(sample_list)}, validity rate wd check: {valid_wd_rate:.4f}")
            if valid_wd_rate <= 0.001 and step >= 6000:
                self.early_stop_count += 1
            if valid_wd_rate > self.max_validity:
                self.max_validity = valid_wd_rate
                self.early_stop_count = 0

            wandb_metrics= {'eval/valid': valid_wd_rate, 'eval/early_stop_count': self.early_stop_count}


            if valid_wd_rate >= 0.80:
                gen_smile_list = [Chem.MolToSmiles(mol) for mol in sample_list]
                if self.data_name in ['moses']:
                    tmp_metrics = get_all_metrics(gen=gen_smile_list, test=self.test_smiles, train=None, device=self.device,
                                index=['SNN','Scaf','FCD'], n_jobs=8)
                else:
                    tmp_metrics = get_all_metrics(gen=gen_smile_list, test=self.test_smiles, train=None, device=self.device,
                                index=['FCD'], n_jobs=8)

                metrics.update(tmp_metrics)

                if self.data_name in ['moses']:
                    self.logger.log(f"step: {step}, n_mol: {len(sample_list)}, fcd_test: {metrics['FCD']:.4f}, SNN: {metrics['SNN']:.4f}, Scaf: {metrics['Scaf']:.4f}")
                    wandb_metrics.update({'eval/SNN': float(metrics['SNN']), 'eval/Scaf': float(metrics['Scaf']), 'eval/FCD': float(metrics['FCD']),
                                      'eval/overall_score': float(metrics['SNN'] + 10 * metrics['Scaf'] - metrics['FCD']/5.0)})
                else:
                    self.logger.log(f"step: {step}, n_mol: {len(sample_list)}, fcd_test: {metrics['FCD']:.4f}")
                    wandb_metrics.update({'eval/FCD': float(metrics['FCD'])})
                if 'FCD' in metrics and metrics['FCD'] < self.min_FCD:
                    self.min_FCD = metrics['FCD']
                    self.early_stop_count = 0
                    wandb.summary['min_FCD'] = float(self.min_FCD)
                    wandb.summary['min_FCD_step'] = int(step)
                if 'SNN' in metrics and metrics['SNN'] > self.max_SNN:
                    self.max_SNN = metrics['SNN']
                    self.early_stop_count = 0
                    wandb.summary['max_SNN'] = float(self.max_SNN)
                    wandb.summary['max_SNN_step'] = int(step)
                if 'Scaf' in metrics and metrics['Scaf'] > self.max_Scaf:
                    self.max_Scaf = metrics['Scaf']
                    self.early_stop_count = 0
                    wandb.summary['max_Scaf'] = float(self.max_Scaf)
                    wandb.summary['max_Scaf_step'] = int(step)

            wandb.log(wandb_metrics, step=step)

        except Exception as e:
            print(f"An error occurred: {e}")

        return metrics


    def _generic_eval_fn(self, step, min_error, sample_con):
        result_dict = None
        if self.mode == 'higher-order' and self.ou_ckpt_path is None:
            return result_dict
        if not sample_con:
            self.early_stop_count += 1


        if self.data_name in ['sbm']:
            sample_size = self.config.eval.batch_size
        else:
            sample_size = min(len(self.test_graph_list), self.config.eval.batch_size)

        if self.mode == 'OU':
            _, _, bond_sample = self.sampling_fn(self.score_model, self.ou_mu)
        elif self.mode == 'higher-order':
            ho_x, ho_la, ho_adj = self.sampling_fn(self.score_model, self.mu)
            ou_mu = {'ho_x': ho_x, 'ho_la': ho_la, 'ho_adj': ho_adj}
            ou_mu.update(self.ou_mu)
            _, _, bond_sample = self.ou_sampling_fn(self.ou_score_model, ou_mu)

        adjs = bond_sample[:sample_size]
        gen_graph_list = graph_utils.tensor2nxgraphs(adjs)
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, self.pl)
        ave_error = sum(result_dict.values()) / len(result_dict)
        result_dict['OU-ave'] = ave_error
        if ave_error < min_error:
            min_error = ave_error
            self.early_stop_count = 0

        if self.data_name in ['sbm'] and ave_error > 0.18 and step > 8000:
            self.early_stop_count += 1

        self.logger.log('-' * 30 + f'step={step}(early_stop_count={self.early_stop_count}), Evaluaction results' + '-' * 30)
        for metric, value in result_dict.items():
            self.logger.log(f'{metric}: {value}')

        wandb_metrics = {f'eval/{metric}': float(value) for metric, value in result_dict.items()}
        wandb_metrics['eval/early_stop_count'] = self.early_stop_count
        wandb.log(wandb_metrics, step=step)

        return result_dict




def gen_mu_for_ho_sampling(train_ds, config, device=None):
    """Generate mu for higher-order sampling.

    Returns:
        mu: input for phase-1 sampling.
        ou_mu: combined with phase-1 output and fed into phase-2 sampling.
    """
    rng = torch.Generator()
    rng.manual_seed(config.eval.seed)
    idx = torch.randint(0, len(train_ds), (config.eval.batch_size,), generator=rng)
    ou_masker = data_masker(train_ds.tensors[3][idx], config, device)
    u0 = train_ds.tensors[2][idx].to(device)
    # ho_masker and ho_u are used in phase 1 (higher-order).
    ho_masker = data_masker(train_ds.tensors[7][idx], config, device)
    ho_u = train_ds.tensors[6][idx].to(device)
    mu = {'u0': ho_u, 'masker': ho_masker}
    ou_mu = {'ho_masker': ho_masker, 'u0': u0, 'masker': ou_masker}

    return mu, ou_mu
