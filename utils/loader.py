import os
import torch
import numpy as np
import random
import logging, yaml, socket
from easydict import EasyDict as edict

import torch.optim as optim
from contextlib import contextmanager
from utils.path_manager import PathManager
from utils.debug_utils import setup_logging
logger = logging.getLogger(__name__)


# Do not delete the following packages
from models.utils import _MODELS
from models.ScoreNet import ConScoreNet, ScoreNet



def init_exp(config, env_cfg_path='env_config.yaml'):
    setup_logging()

    if 'PYTHONHASHSEED' not in os.environ:
        logger.info("Please set PYTHONHASHSEED to speed up evaluation.")
    else:
        logger.info((f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}"))

    env_cfg = set_env_from_config(path=os.path.join(PathManager.CFG_ROOT, env_cfg_path))
    PathManager.init(data_name=config.data.name, exp_name=config.exp_name)

    return env_cfg


def set_env_from_config(path="env_config.yaml"):
    """Load optional env overrides from `path`.

    The file is user-local (gitignored); if it does not exist, fall back to the
    tracked `env_config.example.yaml` template, and finally to an empty config.
    """
    hostname = socket.gethostname()
    prefix = hostname.split('.')[0]  # e.g. "godot123" -> "godot"

    if not os.path.exists(path):
        fallback = path.replace("env_config.yaml", "env_config.example.yaml")
        if os.path.exists(fallback):
            logger.info(f"{path} not found, falling back to {fallback}")
            path = fallback
        else:
            logger.info(f"{path} not found, running without env overrides.")
            return edict({})

    logger.info(f"Setting environment variables from {path} ...")
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Apply the "default" block first; do not overwrite variables already set in the environment.
    # `config.get("default") or {}` tolerates an empty mapping (yaml "default:" with no children -> None).
    for k, v in (config.get("default") or {}).items():
        if v is None or k in os.environ:
            continue
        os.environ[k] = v

    # Then apply the host-specific block, if any.
    for key, envs in config.items():
        if key in ("default", "wandb") or not isinstance(envs, dict):
            continue
        if prefix.startswith(key):
            for k, v in envs.items():
                if v is None or k in os.environ:
                    continue
                os.environ[k] = v
            break
    return edict(config)

def load_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed




@contextmanager
def seed_context(seed):
    # Snapshot current RNG state so it can be restored on exit.
    old_pythonhashseed = os.environ.get('PYTHONHASHSEED')
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    random_state = random.getstate()

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        yield
    finally:
        if old_pythonhashseed is not None:
            os.environ['PYTHONHASHSEED'] = old_pythonhashseed
        else:
            os.environ.pop('PYTHONHASHSEED', None)

        torch.set_rng_state(torch_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(random_state)
        # Restore cudnn defaults.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


_cached_device = None


def load_device(force=False):
    global _cached_device
    if _cached_device is not None and not force:
        return _cached_device
    usage_type = os.getenv('USAGE_TYPE', 'memory')
    def get_device_usage(handle, usage_type):
        """Return the requested usage metric for a GPU handle.

        Args:
            handle: NVML device handle.
            usage_type: 'GPU' (utilization), 'memory' (bytes used), or 'mix' (combined score).
        """
        assert usage_type in ['mix', 'GPU', 'memory']
        if usage_type == 'GPU':
            return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        elif usage_type == 'memory':
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used
        elif usage_type == 'mix':
            used_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            gpu_rate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100
            temperature = (pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) - 30)/40
            return used_memory/total_memory + gpu_rate + temperature

    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if visible_devices is not None:
                visible_devices = [int(dev) for dev in visible_devices.split(",")]
            else:
                visible_devices = list(range(pynvml.nvmlDeviceGetCount()))
            print(f"device available (physical GPU ids):{visible_devices}")

            used_list = []
            for i in visible_devices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                usage = get_device_usage(handle, usage_type)
                used_list.append(usage)

            indexes = np.argmin(used_list)
            print("use device:", indexes)
            device = torch.device(f'cuda:{indexes}')

            pynvml.nvmlShutdown()
        except:
            print("Warning: Cannot import pynvml, use default GPU 0")
            device = torch.device('cuda:0')

    else:
        print("CUDA not available!")
        device = torch.device('cpu')

    _cached_device = device
    return device


def load_score_model(config, device):
    """Create the score model."""
    model_name = config.model.type
    score_model = _MODELS[model_name](config)
    score_model = score_model.to(device)
    return score_model


def load_optimizer(config, params):
    """Return an optimizer based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
    return optimizer

def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn