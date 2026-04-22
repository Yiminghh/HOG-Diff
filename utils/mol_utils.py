import torch
import re
import copy
import numpy as np
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, qed
from utils.sascorer import calculateScore

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def adj2graph(adj, sample_nodes):
    """Covert the PyTorch tensor adjacency matrices to numpy array.

    Args:
        adj: [Batch_size, channel, Max_node, Max_node], assume channel=1
        sample_nodes: [Batch_size]
    """
    adj_list = []
    # discretization
    adj[adj >= 0.5] = 1.
    adj[adj < 0.5] = 0.
    for i in range(adj.shape[0]):
        adj_tmp = adj[i, 0]
        # symmetric
        adj_tmp = torch.tril(adj_tmp, -1)
        adj_tmp = adj_tmp + adj_tmp.transpose(0, 1)
        # truncate
        adj_tmp = adj_tmp.cpu().numpy()[:sample_nodes[i], :sample_nodes[i]]
        adj_list.append(adj_tmp)

    return adj_list


def quantize_mol(adjs):
    # Quantize generated molecules [B, 1, N, N]
    adjs = adjs.squeeze(1)
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs = adjs * 3
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return adjs.to(torch.int64).numpy()


def construct_mol(x, A, num_node, atomic_num_list):
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms = atoms[:num_node]

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    if A.ndim == 2:
        adj = A[:num_node, :num_node]
    elif A.shape[0] == 4:
        raise ValueError('Wrong Adj shape.')
    else:
        raise ValueError('Wrong Adj shape.')

    # Iterate over bonded atom pairs (upper triangle of adjacency).
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # remove formal charge for fair comparison with GraphAF, GraphDF, GraphCNF

            # add formal charge to atom: e.g. [O+], [N+], [S+], not support [O-], [N-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)

    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible valency

    Return:
        True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol):
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []

            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])

    return mol, no_correct


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_chemical_validity(mol):
    """
    Check the chemical validity of the mol object. Existing mol object is not modified.

    Args: mol: Rdkit mol object

    Return:
          True if chemically valid, False otherwise
    """

    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def tensor2mol(x_atom, x_bond, num_atoms, data_name, correct_validity=True, largest_connected_comp=True):
    """Construct molecules from the atom and bond tensors.

    Args:
        x_atom: The node tensor [`number of samples`, `maximum number of atoms`, `number of possible atom types`].
        x_bond: The adjacency tensor [`number of samples`, `number of possible bond type`, `maximum number of atoms`,
            `maximum number of atoms`]
        num_atoms: The number of nodes for every sample [`number of samples`]
        atomic_num_list: A list to specify what each atom channel corresponds to.
        correct_validity: Whether to use the validity correction introduced by `MoFlow`.
        largest_connected_comp: Whether to use the largest connected component as the final molecule in the validity
            correction.

    Return:
        The list of Rdkit mol object. The check_chemical_validity rate without check.
    """
    if x_bond.ndim==3 or x_bond.shape[1] == 1:
        x_bond = quantize_mol(x_bond)
    else:
        x_bond = x_bond.cpu().detach().numpy()


    x_atom = x_atom.cpu().detach().numpy()
    num_nodes = num_atoms.cpu().detach().numpy()

    if data_name.lower() == 'qm9':
        atomic_num_list = [6, 7, 8, 9] 
    elif data_name.lower() == 'zinc250k':
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53] 
    elif data_name.lower() == 'guacamol':
        atomic_num_list = [6, 7, 8, 9, 5, 35, 17, 53, 15, 16, 34, 14]
    elif data_name.lower() == 'moses':
        atomic_num_list = [6, 7, 16, 8, 9, 17, 35, 1]
    else:
        raise ValueError(f"Unsupported data name: {data_name}")

    gen_mols, valid_cum = [], []

    for atom_elem, bond_elem, num_node in zip(x_atom, x_bond, num_nodes):
        mol = construct_mol(atom_elem, bond_elem, num_node, atomic_num_list)

        if correct_validity:
            # correct the invalid molecule
            cmol, no_correct = correct_mol(mol)
            valid_cum.append(1 if no_correct else 0)
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
            gen_mols.append(vcmol)
        else:
            gen_mols.append(mol)

    return gen_mols, valid_cum


def penalized_logp(mol):
    """
    Calculate the reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object

    Returns:
        :class:`float`
    """

    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_mol_qed(mol):
    return qed(mol)


def calculate_min_plogp(mol):
    """
    Calculate the reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`float`
    """

    p1 = penalized_logp(mol)
    s1 = Chem.MolToSmiles(mol, isomericSmiles=True)
    s2 = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    p2 = penalized_logp(mol1)
    p3 = penalized_logp(mol2)
    final_p = min(p1, p2)
    final_p = min(final_p, p3)
    return final_p


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048, useChirality=True):
    """
    Calculate the similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule.

    Args:
        mol: Rdkit mol object
        target: Rdkit mol object

    Returns:
        :class:`float`, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target, radius=radius, nBits=nBits,
                                                            useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)


def convert_radical_electrons_to_hydrogens(mol):
    """
    Convert radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Return a new mol object.

    Args:
        mol: Rdkit mol object

    :rtype:
        Rdkit mol object
    """

    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def get_final_smiles(mol):
    """
    Returns a SMILES of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return Chem.MolToSmiles(m, isomericSmiles=True)


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))

        nx_graphs.append(G)
    return nx_graphs



def canonicalize_smiles(smiles):
    """Canonicalize each SMILES string so equivalent molecules map to the same representation."""

    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]
