"""
# evaluation.mol_KL_evaluator
# Code is soucred from guacamol: https://github.com/BenevolentAI/guacamol
"""

import logging
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from numpy import histogram
from scipy.stats import entropy, gaussian_kde

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Mute RDKit logger noise
RDLogger.logger().setLevel(RDLogger.CRITICAL)


# ----------------------- Core generator interface ----------------------- #
class DistributionMatchingGenerator(metaclass=ABCMeta):
    """
    Minimal interface for molecule generators.
    """

    @abstractmethod
    def generate(self, number_samples: int) -> List[str]:
        """Return a list of SMILES strings."""


class MockGenerator(DistributionMatchingGenerator):
    """
    Simple generator wrapper that returns a fixed set of SMILES strings.
    """

    def __init__(self, samples: List[str]):
        self.samples = samples

    def generate(self, number_samples: int) -> List[str]:
        if number_samples > len(self.samples):
            logger.warning(
                "Requested %d samples, but only have %d. Samples will be repeated.",
                number_samples,
                len(self.samples),
            )
            return (self.samples * (number_samples // len(self.samples) + 1))[
                :number_samples
            ]

        return self.samples[:number_samples]


# ----------------------- Basic chemistry helpers ----------------------- #
def remove_duplicates(list_with_duplicates: List[Any]) -> List[Any]:
    seen: Set[Any] = set()
    uniq: List[Any] = []
    for elem in list_with_duplicates:
        if elem not in seen:
            seen.add(elem)
            uniq.append(elem)
    return uniq


def is_valid(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return smiles != "" and mol is not None and mol.GetNumAtoms() > 0


def canonicalize(smiles: str, include_stereocenters: bool = True) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    return None


def canonicalize_list(
    smiles_list: Iterable[str], include_stereocenters: bool = True
) -> List[str]:
    canonicalized = [
        canonicalize(smiles, include_stereocenters) for smiles in smiles_list
    ]
    canonicalized = [s for s in canonicalized if s is not None]
    return remove_duplicates(canonicalized)


def get_random_subset(dataset: List[Any], subset_size: int, seed: Optional[int] = None):
    if len(dataset) < subset_size:
        raise Exception(
            f"Dataset too small to sample subset: {len(dataset)} < {subset_size}"
        )

    rng_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    subset = np.random.choice(dataset, subset_size, replace=False)
    if seed is not None:
        np.random.set_state(rng_state)
    return list(subset)


def get_mols(smiles_list: Iterable[str]):
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            yield mol


def get_fingerprints(mols: Iterable[Chem.Mol], radius=2, length=4096):
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]


def calculate_internal_pairwise_similarities(smiles_list: Iterable[str]) -> np.ndarray:
    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims
    return similarities


def continuous_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(
        np.hstack([X_baseline, X_sampled]).min(),
        np.hstack([X_baseline, X_sampled]).max(),
        num=1000,
    )
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10
    return entropy(P, Q)


def discrete_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    P, bins = histogram(X_baseline, bins=10, density=True)
    P += 1e-10
    Q, _ = histogram(X_sampled, bins=bins, density=True)
    Q += 1e-10
    return entropy(P, Q)


def _calculate_pc_descriptors(
    smiles: str, pc_descriptors: List[str]
) -> Optional[np.ndarray]:
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(pc_descriptors)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array(calc.CalcDescriptors(mol))
    mask = np.isfinite(fp)
    if (mask == 0).sum() > 0:
        logger.warning("%s contains NAN physchem descriptor", smiles)
        fp[~mask] = 0
    return fp


def calculate_pc_descriptors(smiles: Iterable[str], pc_descriptors: List[str]) -> np.ndarray:
    output = []
    for smi in smiles:
        d = _calculate_pc_descriptors(smi, pc_descriptors)
        if d is not None:
            output.append(d)
    return np.array(output)


# ----------------------- Sampling helpers ----------------------- #
def sample_valid_molecules(
    model: DistributionMatchingGenerator, number_molecules: int, max_tries: int = 10
) -> List[str]:
    max_samples = max_tries * number_molecules
    number_already_sampled = 0
    valid: List[str] = []

    while len(valid) < number_molecules and number_already_sampled < max_samples:
        remaining = number_molecules - len(valid)
        samples = model.generate(remaining)
        number_already_sampled += remaining
        valid += [m for m in samples if is_valid(m)]
    return valid


def sample_unique_molecules(
    model: DistributionMatchingGenerator, number_molecules: int, max_tries: int = 10
) -> List[str]:
    max_samples = max_tries * number_molecules
    number_already_sampled = 0
    uniq_list: List[str] = []
    uniq_set: Set[str] = set()

    while len(uniq_list) < number_molecules and number_already_sampled < max_samples:
        remaining = number_molecules - len(uniq_list)
        samples = model.generate(remaining)
        number_already_sampled += remaining
        for smiles in samples:
            canonical_smiles = canonicalize(smiles)
            if canonical_smiles is not None and canonical_smiles not in uniq_set:
                uniq_set.add(canonical_smiles)
                uniq_list.append(canonical_smiles)
    return uniq_list


# ----------------------- Benchmark scaffolding ----------------------- #
class DistributionLearningBenchmarkResult:
    def __init__(
        self,
        benchmark_name: str,
        score: float,
        sampling_time: float,
        metadata: Dict[str, Any],
    ):
        self.benchmark_name = benchmark_name
        self.score = score
        self.sampling_time = sampling_time
        self.metadata = metadata


class DistributionLearningBenchmark:
    def __init__(self, name: str, number_samples: int):
        self.name = name
        self.number_samples = number_samples

    @abstractmethod
    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        ...


class ValidityBenchmark(DistributionLearningBenchmark):
    def __init__(self, number_samples: int):
        super().__init__(name="Validity", number_samples=number_samples)

    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        start_time = time.time()
        molecules = model.generate(number_samples=self.number_samples)
        end_time = time.time()

        if len(molecules) != self.number_samples:
            raise Exception("The model did not generate the correct number of molecules")

        number_valid = sum(1 if is_valid(smiles) else 0 for smiles in molecules)
        validity_ratio = number_valid / self.number_samples
        metadata = {"number_samples": self.number_samples, "number_valid": number_valid}

        return DistributionLearningBenchmarkResult(
            benchmark_name=self.name,
            score=validity_ratio,
            sampling_time=end_time - start_time,
            metadata=metadata,
        )


class UniquenessBenchmark(DistributionLearningBenchmark):
    def __init__(self, number_samples: int):
        super().__init__(name="Uniqueness", number_samples=number_samples)

    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        start_time = time.time()
        molecules = sample_valid_molecules(
            model=model, number_molecules=self.number_samples
        )
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning(
                "The model could not generate enough valid molecules. The score will be penalized."
            )

        unique_molecules = canonicalize_list(
            molecules, include_stereocenters=False
        )
        unique_ratio = len(unique_molecules) / self.number_samples
        metadata = {
            "number_samples": self.number_samples,
            "number_unique": len(unique_molecules),
        }

        return DistributionLearningBenchmarkResult(
            benchmark_name=self.name,
            score=unique_ratio,
            sampling_time=end_time - start_time,
            metadata=metadata,
        )


class NoveltyBenchmark(DistributionLearningBenchmark):
    def __init__(self, number_samples: int, training_set: Iterable[str]):
        super().__init__(name="Novelty", number_samples=number_samples)
        self.training_set_molecules = set(
            canonicalize_list(training_set, include_stereocenters=False)
        )

    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        start_time = time.time()
        molecules = sample_unique_molecules(
            model=model, number_molecules=self.number_samples, max_tries=2
        )
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning(
                "The model could not generate enough unique molecules. The score will be penalized."
            )

        unique_molecules = set(
            canonicalize_list(molecules, include_stereocenters=False)
        )
        novel_molecules = unique_molecules.difference(self.training_set_molecules)
        novel_ratio = len(novel_molecules) / self.number_samples
        metadata = {
            "number_samples": self.number_samples,
            "number_novel": len(novel_molecules),
        }

        return DistributionLearningBenchmarkResult(
            benchmark_name=self.name,
            score=novel_ratio,
            sampling_time=end_time - start_time,
            metadata=metadata,
        )


class KLDivBenchmark(DistributionLearningBenchmark):
    def __init__(self, number_samples: int, training_set: List[str]):
        super().__init__(name="KL divergence", number_samples=number_samples)
        self.training_set_molecules = canonicalize_list(
            get_random_subset(training_set, self.number_samples, seed=42),
            include_stereocenters=False,
        )
        self.pc_descriptor_subset = [
            "BertzCT",
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAliphaticRings",
            "NumAromaticRings",
        ]

    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        start_time = time.time()
        molecules = sample_unique_molecules(
            model=model, number_molecules=self.number_samples, max_tries=2
        )
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning(
                "The model could not generate enough unique molecules. The score will be penalized."
            )

        unique_molecules = set(
            canonicalize_list(molecules, include_stereocenters=False)
        )

        d_sampled = calculate_pc_descriptors(unique_molecules, self.pc_descriptor_subset)
        d_ref = calculate_pc_descriptors(
            self.training_set_molecules, self.pc_descriptor_subset
        )

        kldivs: Dict[str, float] = {}
        for i in range(4):
            kldivs[self.pc_descriptor_subset[i]] = continuous_kldiv(
                X_baseline=d_ref[:, i], X_sampled=d_sampled[:, i]
            )
        for i in range(4, 9):
            kldivs[self.pc_descriptor_subset[i]] = discrete_kldiv(
                X_baseline=d_ref[:, i], X_sampled=d_sampled[:, i]
            )

        ref_sim = calculate_internal_pairwise_similarities(self.training_set_molecules)
        ref_sim = ref_sim.max(axis=1)
        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)
        kldivs["internal_similarity"] = continuous_kldiv(
            X_baseline=ref_sim, X_sampled=sampled_sim
        )

        metadata = {"number_samples": self.number_samples, "kl_divs": kldivs}
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return DistributionLearningBenchmarkResult(
            benchmark_name=self.name,
            score=score,
            sampling_time=end_time - start_time,
            metadata=metadata,
        )


# ----------------------- Public API ----------------------- #
def evaluate_samples(generated_smiles: List[str], reference_smiles: List[str]):
    """
    Run Validity/Uniqueness/Novelty and KL benchmarks on the provided samples.
    """
    model = MockGenerator(generated_smiles)
    number_samples = min(len(generated_smiles), 10000)

    print(f"Evaluating with {number_samples} samples...")
    rel = {}

    validity_result = ValidityBenchmark(number_samples=number_samples).assess_model(model)
    print(f"Validity Metadata: {validity_result.metadata}")

    uniqueness_result = UniquenessBenchmark(number_samples=number_samples).assess_model(model)
    print(f"Uniqueness Metadata: {uniqueness_result.metadata}")

    novelty_result = NoveltyBenchmark(number_samples=number_samples, training_set=reference_smiles).assess_model(model)
    print(f"Novelty Metadata: {novelty_result.metadata}")

    kl_result = KLDivBenchmark(number_samples=number_samples, training_set=reference_smiles).assess_model(model)
    print(f"KL Divergence Score: {kl_result.score:.4f}")
    print(f"KL Metadata: {kl_result.metadata}")
    
    rel = {
        'Validity': validity_result.metadata,
        'Uniqueness': uniqueness_result.metadata,
        'Novelty': novelty_result.metadata,
        'KL div meta': kl_result.metadata,
        'KL div': kl_result.score,
    }
    return rel



