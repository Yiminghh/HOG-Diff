import os
import time
import wandb
import re
import logging
from easydict import EasyDict as edict
from contextlib import contextmanager



# Default WANDB_MODE = disabled when not set explicitly.
WANDB_MODE = os.environ.setdefault("WANDB_MODE", "disabled").lower()
# Global debug flag.
_DEBUG_ = os.environ.setdefault("_DEBUG_", 'true').lower() in  ["1", "true", "yes"]

logger = logging.getLogger(__name__)


class LevelBasedFormatter(logging.Formatter):
    """Bare message at DEBUG/INFO; `[LEVEL-logger] message` at WARNING and above."""
    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "[%(levelname)s-%(name)s] %(message)s"
        return super().format(record)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if _DEBUG_ else logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(LevelBasedFormatter())
    root.addHandler(handler)

    logger.info(f"DEBUG={_DEBUG_}, WANDB_MODE={WANDB_MODE}")


@contextmanager
def Timer(name="Running Time", enabled=_DEBUG_):
    """Context manager timer; prints elapsed seconds only when enabled."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            end = time.perf_counter()
            print(f"{name}: {end - start:.4f} s")


def to_dict(x):
    """Convert EasyDict (or nested EasyDict) to a plain dict so wandb can log it."""
    if isinstance(x, dict):
        return {k: to_dict(v) for k, v in x.items()}
    try:
        if isinstance(x, edict):
            return to_dict(dict(x))
    except Exception:
        pass
    return x


def apply_wandb_config_to_cfg(cfg, mode):
    """Override cfg with nested wandb.config keys (e.g. ``train.lr``) when running a sweep."""
    if wandb.run is None or getattr(wandb.run, "sweep_id", None) is None:
        return cfg
    sci = re.compile(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$')
    for key, val in dict(wandb.config).items():
        if '.' not in key:
            # Only nested keys (like train.lr) are supported; skip flat keys.
            continue
        parts, target = key.split('.'), cfg
        for p in parts[:-1]:
            if hasattr(target, p):
                target = getattr(target, p)
            else:
                target = None; break
        if target is None:
            continue
        last = parts[-1]
        if isinstance(val, str) and sci.match(val):
            try: val = float(val)
            except ValueError: pass
        setattr(target, last, val)

    cfg.sde.bond.num_scales = cfg.sde.atom.num_scales
    cfg.exp.plot = False
    if cfg.sampling.n_steps <= 0:
        cfg.sampling.corrector = 'None'

    if mode == 'higher-order':
        cfg.sde.bond.beta_min = cfg.sde.atom.beta_min
        cfg.sde.bond.beta_max = cfg.sde.atom.beta_max
    elif mode == 'OU':
        cfg.sde.bond.schedule = cfg.sde.atom.schedule
        cfg.sde.bond.max_sigma = cfg.sde.atom.max_sigma
    return cfg



class CheckpointManager:
    """Pareto-front checkpoint tracker with threshold filtering and size cap.

    Only non-dominated checkpoints (no other ckpt is better on every monitored metric)
    are kept. When the front exceeds ``max_size``, the worst entries by average per-metric
    rank are evicted.

    ``_ALL_METRICS_INFO`` maps each supported metric to its direction:
        True  -> higher-is-better (e.g. validity, SNN, Scaf)
        False -> lower-is-better  (e.g. loss, FCD, NSPDK)
    """

    _ALL_METRICS_INFO = {
        'loss': False,
        'NSPDK': False,
        'FCD': False,
        'validity': True,
        'SNN': True,
        'Scaf': True,
    }

    def __init__(self, thresholds: dict = dict(), max_size: int = 5):
        """
        Args:
            thresholds: per-metric cutoffs. Any checkpoint failing a threshold is
                dropped immediately. The keys of ``thresholds`` also define which
                metrics participate in Pareto comparison (must be a subset of
                ``_ALL_METRICS_INFO.keys()``).
            max_size: maximum number of entries kept on the Pareto front.
        """
        if not thresholds.keys() <= self._ALL_METRICS_INFO.keys():
            raise ValueError(f"[CheckpointManager] metrics_to_monitor must be a subset of: {set(self._ALL_METRICS_INFO.keys())}")

        self.max_size = max_size
        self.thresholds = thresholds
        self.metrics_info = {key: self._ALL_METRICS_INFO[key] for key in thresholds.keys()}

        self.pareto_list = []

    def _meets_thresholds(self, metrics: dict) -> bool:
        """Return True if ``metrics`` satisfies every threshold in ``self.thresholds``."""
        for key, thresh in self.thresholds.items():
            val = metrics.get(key)
            if val is None:
                continue
            rev = self.metrics_info[key]
            if (val < thresh and rev) or (val > thresh and not rev):
                return False

        return True

    def _is_dominated(self, m1: dict, m2: dict) -> bool:
        """Return True iff m1 is dominated by m2 on the monitored metrics."""
        strictly_better = False

        for key, rev in self.metrics_info.items():
            v1, v2 = m1.get(key), m2.get(key)
            if v1 is None or v2 is None:
                continue

            if rev:
                if v2 < v1:
                    return False
                if v2 > v1:
                    strictly_better = True
            else:
                if v2 > v1:
                    return False
                if v2 < v1:
                    strictly_better = True

        return strictly_better

    def _compute_average_ranks(self) -> dict:
        """Return {path: avg_rank_over_monitored_metrics} for entries on the front."""
        n = len(self.pareto_list)
        if n == 0:
            return {}

        metric_values = {key: [] for key in self.metrics_info}
        for entry in self.pareto_list:
            for key in self.metrics_info:
                metric_values[key].append(entry['metrics'].get(key, 0.0))

        ranks = {entry['path']: [] for entry in self.pareto_list}

        for key, rev in self.metrics_info.items():
            values = metric_values[key]
            if rev:
                sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=True)
            else:
                sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=False)

            key_ranks = [0] * n
            for rank, idx in enumerate(sorted_indices, start=1):
                key_ranks[idx] = rank

            for i, entry in enumerate(self.pareto_list):
                ranks[entry['path']].append(key_ranks[i])

        avg_ranks = {
            entry['path']: sum(ranks[entry['path']]) / len(ranks[entry['path']])
            for entry in self.pareto_list
        }
        return avg_ranks

    def update(self, ckpt_path: str, metrics: dict) -> (bool, set):
        """Update the Pareto front with ``(ckpt_path, metrics)``.

        Returns:
            (should_save, remove_set):
                should_save — whether the new checkpoint made it onto the front.
                remove_set  — paths of checkpoints that should now be deleted.
        """
        if not self._meets_thresholds(metrics):
            return False, set()

        for entry in self.pareto_list:
            if self._is_dominated(metrics, entry['metrics']):
                return False, set()

        survivors = [{'metrics': metrics.copy(), 'path': ckpt_path}]
        removed_paths = []
        for entry in self.pareto_list:
            if self._is_dominated(entry['metrics'], metrics):
                removed_paths.append(entry['path'])
            else:
                survivors.append(entry)
        self.pareto_list = survivors


        if len(self.pareto_list) > self.max_size:
            avg_ranks = self._compute_average_ranks()
            sorted_paths = sorted(avg_ranks.keys(), key=lambda p: avg_ranks[p])
            keep_paths = set(sorted_paths[: self.max_size])

            for entry in list(self.pareto_list):
                if entry['path'] not in keep_paths:
                    removed_paths.append(entry['path'])
            self.pareto_list = [e for e in self.pareto_list if e['path'] in keep_paths]

        keep_set = {entry['path'] for entry in self.pareto_list}
        should_save = (ckpt_path in keep_set)
        remove_set = set(removed_paths) - keep_set

        return should_save, remove_set
