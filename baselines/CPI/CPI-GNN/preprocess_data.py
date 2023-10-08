from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def extract_fingerprints_exec(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # if (len(atoms) == 1) or (radius == 0):
    #     fingerprints = [fingerprint_dict[a] for a in atoms]
    #
    # else:
    nodes = atoms
    i_jedge_dict = i_jbond_dict

    # for _ in range(radius):
    #     """Update each node ID considering its neighboring nodes and edges
    #     (i.e., r-radius subgraphs or fingerprints)."""
    fingerprints = []
    for i, j_edge in i_jedge_dict.items():
        neighbors = [(nodes[j], edge) for j, edge in j_edge]
        fingerprint = (nodes[i], tuple(sorted(neighbors)))
        fingerprints.append(fingerprint_dict[fingerprint])
    nodes = fingerprints

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":
    import pandas as pd
    import pickle as pkl

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    df_train = pd.read_csv('./data/train1.csv')
    df_test = pd.read_csv('./data/test1.csv')
    smile_list = set(list(df_train['0']) + list(df_test['0']))
    seq_list = set(list(df_train['1']) + list(df_test['1']))

    smile_dict = {}
    total = 0
    for smile in smile_list:
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius=0)
        adjacency = create_adjacency(mol)
        smile_dict[smile] = (fingerprints, adjacency)


    print(total)
    seq_dict = {}
    for seq in seq_list:
        words = split_sequence(seq, ngram=3)
        seq_dict[seq] = words

    with open('./data/smile_dict.pkl', 'wb') as file:
        pkl.dump(smile_dict, file)

    with open('./data/seq_dict.pkl', 'wb') as file:
        pkl.dump(seq_dict, file)

    with open('./data/fingerprint_dict.pkl', 'wb') as file:
        pkl.dump(dict(fingerprint_dict), file)

    with open('./data/word_dict.pkl', 'wb') as file:
        pkl.dump(dict(word_dict), file)




