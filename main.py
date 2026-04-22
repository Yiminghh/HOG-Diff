import os
import trainer, sampler
from absl import app, flags
from utils.config_paraser import load_config
from utils.logger import Logger
from utils.dataloader import dataloader
from utils.loader import init_exp
from utils.path_manager import PathManager
import wandb
import torch
from datetime import datetime
from utils.debug_utils import to_dict, apply_wandb_config_to_cfg
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'guacamol.yaml', 'Configuration.')
flags.DEFINE_string('exp_name', "main-test", 'Experiment name')
flags.DEFINE_enum('mode', 'train_OU', ['train_ho', 'train_OU', 'sample'], 'Type of program.')
flags.DEFINE_string('sample_prior', 'None', 'Prior for sampling: None or mu')
flags.DEFINE_string('ckpt', None, 'ckpt')
flags.DEFINE_string('ckpt_meta', None, 'ckpt')
flags.DEFINE_string('wandb_project', 'HOG-Diff-moses', 'wandb project name')

def main(argv):
    """Main entry: run training or sampling pipeline."""

    hconfig = load_config(FLAGS.config, FLAGS.exp_name, mode='higher-order')
    ouconfig = load_config(FLAGS.config, FLAGS.exp_name, mode='OU')
    env_cfg = init_exp(ouconfig)


    wandb_entity = env_cfg.get('wandb', {}).get('entity') if isinstance(env_cfg, dict) else getattr(getattr(env_cfg, 'wandb', None), 'entity', None)
    wandb.init(
        entity=wandb_entity,
        project=FLAGS.wandb_project,
        name=f"{os.getenv('WANDB_SWEEP', '')}{datetime.now().strftime('%y%m%d-%H:%M%S')}",
        dir=PathManager.WANDB_LOG_ROOT,
        config=to_dict({'hconfig': to_dict(hconfig), 'ouconfig': to_dict(ouconfig)}),
    )
    if FLAGS.ckpt is not None:
        hconfig.ckpt = ouconfig.ckpt = FLAGS.ckpt
    if FLAGS.ckpt_meta is not None:
        hconfig.ckpt_meta = ouconfig.ckpt_meta = FLAGS.ckpt_meta


    if FLAGS.mode == 'train_ho':
        hconfig = apply_wandb_config_to_cfg(hconfig, mode='higher-order')
        train_logger = Logger(hconfig, is_train=True, show_exc=True)
        htrainer = trainer.Trainer(hconfig, train_logger, mode='higher-order')
        htrainer.train()
    elif FLAGS.mode == 'train_OU':
        ouconfig = apply_wandb_config_to_cfg(ouconfig, mode='OU')
        train_logger = Logger(ouconfig, is_train=True, show_exc=True)
        outrainer = trainer.Trainer(ouconfig, train_logger, mode='OU')
        outrainer.train()
    elif FLAGS.mode == 'sample':
        sample_logger = Logger(ouconfig, is_train=False, show_exc=True)

        hsampler = None
        if FLAGS.sample_prior == 'None' or FLAGS.sample_prior is None:
            hsampler = sampler.Sampler(hconfig, sample_logger, mode='higher-order')
            mu, _ = hsampler.sample()
        elif FLAGS.sample_prior == 'mu':
            # Use higher-order information directly as the phase-2 prior.
            mu = sampler.get_mu_from_ho_directedly(dataloader(ouconfig, mode='OU'), ouconfig)


        ousampler = sampler.Sampler(ouconfig, sample_logger, mode='OU')
        out_list, result_dict = ousampler.sample(mu_list=mu)
        sample_logger.rename_log(result_dict)

        if getattr(ousampler.config.exp, 'plot', False):
            chain_rel = {'OU': ousampler.chain_rel}
            if hsampler is not None:
                chain_rel['higher-order'] = hsampler.chain_rel
            chain_rel_path = os.path.join(
                PathManager.PRO_ROOT, 'analysis', 'vis_chains', ouconfig.data.name,
                f"chain_rel({datetime.now().strftime('%Y%m%d_%H%M%S')}).pth"
            )
            os.makedirs(os.path.dirname(chain_rel_path), exist_ok=True)
            torch.save(chain_rel, chain_rel_path)
            print(f"Saved chain_rel to {os.path.abspath(chain_rel_path)}")

    print("HOG-Diff Finished!")
    wandb.finish()



if __name__ == '__main__':
    app.run(main)
