# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
import json
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse


def atom_features(atom):
    # Generating initial atomic descriptors based on atomic properties.
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """
    Convert a drug smile into topo graph (i.e., feature matrix and adjacency matrix )
    :param smile: smile string
    :return: nodenum, feature matrix, and adjacency matrix(sparse).
    """
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    assert len(edge_index) != 0

    return c_size, features, edge_index


def dic_normalize(dic):
    """
    Feature normalization
    :param dic: A dict for describing residue feature
    :return: Normalizied feature
    """
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# predefined resdiue type space
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
# predefined aliphatic resdiue
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
# predefined aromatic resdiue
pro_res_aromatic_table = ['F', 'W', 'Y']
# predefined polar neutral resdiue
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
# predefined acidic resdiue
pro_res_acidic_charged_table = ['D', 'E']
# predefined basic resdiue
pro_res_basic_charged_table = ['H', 'K', 'R']

# predefined resdiue weigh space
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

#Dissociation constant for the –COOH group
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
#Dissociation constant for the –NH3 group
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
#Dissociation constant for any other group in the molecule
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
#pH at the isoelectric point
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

#Hydrophobicity of the residue(pH = 2)
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

#Hydrophobicity of the residue (pH = 7)
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

#normalization
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    # Generating initial residual descriptors based on atomic properties.
    if residue not in pro_res_table:
        residue = 'X'
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    """
    Generate feature descriptors for all residues in one protein
    :param pro_seq: protein sequence
    :return: the initial residue feature descriptors (33 dim) for pro_seq
    """
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding_unk(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def seq_to_graph(pro_id, seq, db):
    """
    Convert a protein sequence into topo graph (i.e., feature matrix and adjacency matrix )
    :param pro_id: the UniProt id of protein
    :param seq: protein sequence
    :param db: dataset name
    :return: nodenum, feature matrix, and adjacency matrix(sparse).
    """
    sparse_edge_matrix = np.load(f'data/{db}/contact_map/{pro_id}.npy') # reading contact map
    edge_index = np.argwhere(sparse_edge_matrix == 1).transpose(1, 0)

    c_size = len(seq)
    features = seq_feature(seq)
    return c_size, features, edge_index



def data_split(dataset):
    """
    Make dataset spliting and Convert the original data into csv format.
    No value is returned, but the corresponding csv file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    if dataset == 'Human':
        print('convert human data into 5-fold sub-dataset !')
        df = pd.read_table('data/Human/Human.txt', sep=' ', header=None)
        print(len(list(df[0])))

        with open('data/Human/invalid_seq.pkl','rb') as file:
            invalid_seq = pickle.load(file)
        with open('data/Human/invalid_smile.pkl', 'rb') as file:
            invalid_smile = pickle.load(file)

        #Filter out drug molecules or proteins that cannot be graphed in Human.
        for i in invalid_smile:
            index = df[df[0] == i].index.tolist()
            df = df.drop(index)
        print(len(list(df[0])))
        for i in invalid_seq:
            index = df[df[1] == i].index.tolist()
            df = df.drop(index)
        print(len(list(df[0])))

        # Five-fold splitting
        portion = int(0.2 * len(df[0]))
        for fold in range(5):
            if fold < 4:
                df_test = df.iloc[fold * portion:(fold + 1) * portion]
                df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
            if fold == 4:
                df_test = df.iloc[fold * portion:]
                df_train = df.iloc[:fold * portion]
            assert (len(df_test) + len(df_train)) == len(df)
            df_test.to_csv(f'data/Human/test{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
            df_train.to_csv(f'data/Human/train{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
    else:
        print('convert data from DeepDTA for ', dataset)
        # Read the data from the raw file.
        fpath = 'data/' + dataset + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e ]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
        drugs = []
        prots = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
            drugs.append(lg)
        for t in proteins.keys():
            prots.append(proteins[t])

        if dataset == 'davis':
            affinity = [-np.log10(y/1e9) for y in affinity] # Perform a negative logarithmic transformation on the affinity data in the Davis dataset.

        affinity = np.asarray(affinity)
        opts = ['train','test']
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  #Filter out invalid affinity data.
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
            with open('data/' + dataset + '/' + opt + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')
        print('\ndataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))

def construct_graph(args):
    """
    Construct topological graph for protein (pro_data) and drug (mol_data), respectively.
    No value is returned, but the corresponding graph data file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    print('Construct graph for ', args.dataset)
    ## 1. generate drug graph dict.
    compound_iso_smiles = []
    if args.dataset == 'Human':
        opts = ['train1', 'test1']
    else:
        opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv(f'data/{args.dataset}/' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('drug graph is constructed successfully!')

    ## 2.generate protein graph dict.
    seq_dict = {}
    with open(f'data/{args.dataset}/{args.dataset}_dict.txt', 'r') as file:
        for line in file.readlines():
            line = line.lstrip('>').strip().split('\t')
            seq_dict[line[1]] = line[0]
    seq_graph = {}
    for pro_id, seq in seq_dict.items():
        g = seq_to_graph(pro_id, seq, args.dataset)
        seq_graph[seq] = g
    print('protein graph is constructed successfully!')

    ## 3. Serialized graph data
    with open(f'data/{args.dataset}/mol_data.pkl', 'wb') as smile_file:
        pickle.dump(smile_graph, smile_file)
    with open(f'data/{args.dataset}/pro_data.pkl', 'wb') as seq_file:
        pickle.dump(seq_graph, seq_file)


def fold_split_for_davis():
    """ The train set of davis was split into 5 subsets for finetuning hyper-parameter."""
    df = pd.read_csv('data/davis/train.csv')
    portion = int(0.2 * len(df['affinity']))

    for fold in range(5):
        if fold < 4:
            df_test = df.iloc[fold * portion:(fold + 1) * portion]
            df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
        if fold == 4:
            df_test = df.iloc[fold * portion:]
            df_train = df.iloc[:fold * portion]
        assert (len(df_test) + len(df_train)) == len(df)
        df_test.to_csv(f'data/davis/5 fold/test{fold + 1}.csv', index=False)
        df_train.to_csv(f'data/davis/5 fold/train{fold + 1}.csv', index=False)


def main(args):
    data_split(args.dataset)
    construct_graph(args)
    # fold_split_for_davis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'davis', help='dataset name',choices=['davis','kiba','Human'])
    args = parser.parse_args()
    main(args)
