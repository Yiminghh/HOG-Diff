# evaluation.mol_nspdk_evaluator

### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py

import os.path

import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import pairwise_kernels
from evaluation.eden import vectorize
import logging
from utils.mol_utils import mols_to_nx
from rdkit import Chem
from utils.path_manager import PathManager
logger = logging.getLogger(__name__)




# Lazily compute the reference vector and cache it for reuse.
_ref_vec = None
_X_mean  = None

def _prepare_reference(test_smiles):
    global _ref_vec, _X_mean
    if _ref_vec is not None and _X_mean is not None:
        return
    seed_env = os.environ.get("PYTHONHASHSEED", None)
    if seed_env is None:
        logger.info("Setting 'PYTHONHASHSEED' can accelerate NSPDK computing by avoid re-computing reference vectors.")

    ref_cache_path = os.path.join(PathManager.DATA_CACHE_DIR,  f'ref_vec-hash{seed_env}.npz')
    Xmean_cache_path   = os.path.join(PathManager.DATA_CACHE_DIR,  f'ref_kernel_X_mean-hash{seed_env}.npy')

    if seed_env is not None and os.path.exists(ref_cache_path) and os.path.exists(Xmean_cache_path):
        _ref_vec = load_npz(ref_cache_path)
        _X_mean = np.load(Xmean_cache_path).item()
    else:
        mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        mol_ref_list = mols_to_nx(mols)

        _ref_vec = vectorize(mol_ref_list, complexity=4, discrete=True)
        X = pairwise_kernels(_ref_vec, None, metric='linear', n_jobs=20)
        _X_mean = np.average(X)

        save_npz(ref_cache_path, _ref_vec)
        # np.save stores a 0-d array; use .item() on load to recover the scalar.
        np.save(Xmean_cache_path, np.array(_X_mean))


def nspdk_eval_fn(mol_pred_list, test_smiles=None, n_jobs = 20):
    if test_smiles is not None:
        _prepare_reference(test_smiles)

    mol_pred_list = mols_to_nx(mol_pred_list)
    mol_pred_list = [G for G in mol_pred_list if not G.number_of_nodes() == 0]
    pred = vectorize(mol_pred_list, complexity=4, discrete=True)

    Y = pairwise_kernels(pred, None, metric='linear', n_jobs=n_jobs)
    Z = pairwise_kernels(_ref_vec, pred, metric='linear', n_jobs=n_jobs)

    return _X_mean + np.average(Y) - 2 * np.average(Z)
