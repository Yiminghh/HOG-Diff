import inspect
import os
import logging
import socket


logger = logging.getLogger(__name__)

class PathManagerMeta(type):
    """Metaclass that prevents access to dynamic PathManager attributes before init()."""
    def __getattribute__(cls, name):
        allowed = {'__class__', '__name__', '__module__', '__doc__',
                   'init', '_initialized',
                   'PRO_ROOT', 'CFG_ROOT', }
        if name not in allowed and not super().__getattribute__('_initialized'):
            raise RuntimeError("PathManager not initialized. Call PathManager.init(exp_name, data_name) first.")
        return super().__getattribute__(name)


class PathManager(metaclass=PathManagerMeta):
    """Central registry for project paths.

    Static `_ROOT` paths are always accessible:
        PRO_ROOT, CFG_ROOT, DTA_ROOT, CKPT_ROOT, WANDB_LOG_ROOT.

    Dynamic `_DIR` paths require a prior call to PathManager.init(exp_name, data_name):
        DATA_DIR, CKPT_DIR, LOGS_DIR, etc.
    """
    HOSTNAME = socket.gethostname().split('.')[0]
    PRO_ROOT = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
    CFG_ROOT = os.path.join(PRO_ROOT, 'configs')


    EXP_NAME = None
    DATA_NAME = None
    _initialized = False

    @classmethod
    def init(cls, data_name, exp_name=None):
        if cls._initialized:
            assert exp_name == cls.EXP_NAME and data_name == cls.DATA_NAME,\
                f"PathManager already initialized with data_name={cls.DATA_NAME}, exp_name={cls.EXP_NAME}, cannot reset to {exp_name}"
            return
        cls._initialized = True
        cls.DATA_NAME = data_name
        cls.EXP_NAME = exp_name
        suffix = [exp_name] if exp_name else []

        # Root directories can be overridden via environment variables.
        cls.DTA_ROOT = os.getenv('DATA_ROOT', os.path.join(cls.PRO_ROOT, 'data'))
        cls.CKPT_ROOT = os.getenv('CKPT_ROOT', os.path.join(cls.PRO_ROOT, 'checkpoints'))
        cls.WANDB_LOG_ROOT = os.getenv('WANDB_LOG_ROOT', os.path.join(cls.PRO_ROOT, 'wandb_logs'))

        cls.DATA_DIR = os.path.join(cls.DTA_ROOT, cls.DATA_NAME)
        cls.DATA_RAW_DIR = os.path.join(cls.DATA_DIR, 'raw')
        cls.DATA_PROCESSED_DIR = os.path.join(cls.DATA_DIR, 'processed')
        cls.DATA_CACHE_DIR = os.path.join(cls.DATA_DIR, 'cache')

        cls.CKPT_DIR = os.path.join(cls.CKPT_ROOT, cls.DATA_NAME, *suffix)

        # Create every *_DIR / *_ROOT path that exists as a class attribute.
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR') or attr_name.endswith('_ROOT'):
                dir_path = getattr(cls, attr_name)
                if isinstance(dir_path, str):
                    os.makedirs(dir_path, exist_ok=True)

        logger.info(f"[INFO] Running exp for {data_name}, Host: {cls.HOSTNAME}, Root dir: {cls.PRO_ROOT}, Exp_name: {exp_name}")
