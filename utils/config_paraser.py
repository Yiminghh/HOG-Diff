import yaml
from easydict import EasyDict as edict
from utils.file_utils import get_config_path
from fractions import Fraction
from datetime import datetime
import pytz

def load_config(config_name, exp_name=None, mode='classical'):
    config_path = get_config_path(config_name)
    print(f"Load {mode} config from {config_path}")
    full_config = edict(yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

    if exp_name is not None:
        full_config.exp_name = exp_name

    if mode in ['classical', 'higher-order']:
        # Drop entries whose key starts with "OU" (used only in OU mode).
        full_config = edict({k: v for k, v in full_config.items() if not k.startswith('OU')})
    elif mode in ['OU']:
        # Rename "OU*" keys to their base name so they take over the config.
        ouconfig = edict({k.replace('OU', ''): v for k, v in full_config.items() if k.startswith('OU')})
        full_config.update(ouconfig)

    edge_th = full_config.model.get('edge_th')
    if isinstance(edge_th, str):
        full_config.model.edge_th = float(Fraction(edge_th))
    if full_config.sampling.n_steps <=0:
        full_config.sampling.corrector = 'None'

    return full_config


def get_curr_time():
    return datetime.now(pytz.timezone('Europe/London')).strftime('%b%d(%H-%M-%S)')
