# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年08月18日
"""
import Bio.PDB
import numpy as np
import os
from tqdm import tqdm
import argparse

# Mapping between three-letter and one-letter forms of amino acid residues.
aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
    }

def get_center_atom(residue):
    # Identifying the central atom of a residue
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    c_atom1 = get_center_atom(residue_one)
    c_atom2 = get_center_atom(residue_two)
    diff_vector  = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    residue_seq = ''
    residue_len = 0
    for row, residue_one in enumerate(chain_one):
        hetfield = residue_one.get_id()[0]
        hetname = residue_one.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            atom_ = get_center_atom(residue_one)
            residue_len = residue_len + 1
            residue_seq = residue_seq + aa_codes[hetname]  # Extracting protein sequences
    dist_matrix = np.zeros((residue_len, residue_len), dtype=float)
    x = -1
    for residue_one in chain_one:
        y = -1
        hetfield1 = residue_one.get_id()[0]
        hetname1 = residue_one.get_resname()
        if hetfield1 == ' ' and hetname1 in aa_codes.keys():
            x = x + 1
            for residue_two in chain_two:
                hetfield2 = residue_two.get_id()[0]
                hetname2 = residue_two.get_resname()
                if hetfield2 == ' ' and hetname2 in aa_codes.keys():
                    y = y + 1
                    dist_matrix[x, y]= calc_residue_dist(residue_one, residue_two) # Computing residue contact map.
    for i in range(residue_len):
        dist_matrix[i,i] = 100
    return dist_matrix,residue_seq


def calc_contact_map(pdb_id,chain_id):
    """
    Calculate residue contact maps from protein structure files.
    :param pdb_id: Name of the structure file
    :param chain_id: Protein chain ID, usually chain A
    :return: protein contact map and protein sequence
    """
    pdb_path = pdb_id + '.pdb'
    structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
    model = structure[0]
    dist_matrix,res_seq = calc_dist_matrix(model[chain_id], model[chain_id])
    contact_map = (dist_matrix < 8.0).astype(int)
    return contact_map,res_seq


def main(args):
    pid_list = []
    files = os.listdir(args.input_path)
    for file in files:
        pid_list.append(file.strip().split('.')[0])

    for pid in tqdm(pid_list):
        contact_map, sequence = calc_contact_map(args.input_path + pid, 'A')
        np.save(args.output_path + f'{pid}.npy', contact_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,  help='the directory for storing PDB structure files ')
    parser.add_argument('--output_path', type=str, help='the directory for saving contact map files')
    parser.add_argument('--chain_id', type=str, default = 'A', help='protein chain id')
    args = parser.parse_args()
    main(args)

