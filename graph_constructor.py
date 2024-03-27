# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2023-12-18 16:08:45
# @Last Modified by:   Rexmo
# @Last Modified time: 2023-12-20 17:32:32
import os
import pickle
import dgl
import torch
import numpy as np
import pandas as pd
from functools import partial
from itertools import repeat
from scipy import sparse


import warnings
warnings.filterwarnings("ignore")



warnings.filterwarnings('ignore')
# converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
AAMAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J'}
Standard_AAMAP = {'HIS': 'H', 'ASP': 'D', 'ARG': 'R', 'PHE': 'F', 'ALA': 'A', 'CYS': 'C', 'GLY': 'G', 'GLN': 'Q',
                  'GLU': 'E', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'SER': 'S', 'TYR': 'Y', 'THR': 'T',
                  'ILE': 'I', 'TRP': 'W', 'PRO': 'P', 'VAL': 'V'}
standd_aa_thr = list(Standard_AAMAP.keys())
standd_aa_one = list(Standard_AAMAP.values())
AABLOSUM62 = pd.read_csv('/home/jingxuan/param/AAD/data/AABLOSUM62.csv').iloc[:, 1:]


def struc2seq(struc):
    """
    :param struc: pdb object generated using parsePDB function
    :return: sequence for each struc
    """
    seq = ''
    if struc:
        for _ in struc.iterResidues():
            try:
                letter = standd_aa_one[standd_aa_thr.index(_.getResname())]  # standard amino acids
            except:
                letter = 'U'  # non-standard amino acids
            seq = seq + letter
    else:
        seq = None
    return seq

def get_interact_dict(arr):
    interact_dict = {}
    for i in range(len(arr[0])):
        if arr[0][i] not in interact_dict:
            interact_dict[arr[0][i]] = [arr[1][i]]
        else:
            interact_dict[arr[0][i]].append(arr[1][i])
    return interact_dict

def get_residue_center(protein):
    res_geo_center = []
    for res in protein.iterResidues():
        res_geo_center.append(res.getCoords().mean(axis=0))
    res_geo_center = np.array(res_geo_center)
    return res_geo_center

def ligand_graph(mol, add_self_loop=False):
    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)
    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)
    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)
    return g, src_ls, dst_ls


def pocket_sequence(pocket):
    """
    :param pocket: pdb object generated using parsePDB function
    :return: sequence for each pocket
    """
    # pocket = parsePDB(pocket_file)
    seq = ''
    if pocket:
        for _ in pocket.iterResidues():
            try:
                letter = standd_aa_one[standd_aa_thr.index(_.getResname())]  # standard amino acids
            except:
                letter = 'U'  # non-standard amino acids
            seq = seq + letter
    else:
        seq = None
    return seq


def construct_complete_graph_from_sequence(sequence, add_self_loop=False):
    """Construct a complete graph using sequence of protein pocket

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    sequence : sequence of protein pocket
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the protein pocket graph
    """
    num_residues = len(sequence)
    edge_list = []
    for i in range(num_residues):
        for j in range(num_residues):
            if i != j or add_self_loop:
                edge_list.append((i, j))
    g = dgl.DGLGraph(edge_list)

    return g


def construct_graph_from_sequence(sequence, dev='cpu'):
    """Construct a graph using sequence"""

    num_residues = len(sequence)
    g = dgl.DGLGraph().to(dev)
    g.add_nodes(num_residues)
    return g





def cdr_sequence_featurizer_esm(esm_model, esm_batch_converter, sequence, device):
    """
    :param sequence: sequence information
    :return:
    """
    data = [
        ("seq", sequence),
    ]
    # print(esm_batch_converter)
    batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
    # think it adds beginning and end tokens in the conversion process: ex) seq length 16 --> 18

    batch_tokens = batch_tokens.to(device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    out_temp = results["representations"][33]
    ndata = out_temp[:,1:-1,:].squeeze()
    return ndata


def write_jobs(jobs, graph_dic_path, path_marker):
    for job in jobs:
        dic = job.get()
        if dic is not None:
            with open(graph_dic_path + path_marker + str(dic['key']), 'wb') as f:
                pickle.dump(dic, f)


def collate_fn(data_batch):
    gl_, gl_mask_, gp_, gp_mask_, glp_, y_, key_, info_ = map(list, zip(*data_batch))
    bgl = dgl.batch(gl_)
    bgl_mask = torch.cat(gl_mask_)
    bgp = dgl.batch(gp_)
    bgp_mask = torch.cat(gp_mask_)
    bglp = dgl.batch(glp_)
    y = torch.unsqueeze(torch.stack(y_, dim=0), dim=-1)
    return bgl, bgl_mask, bgp, bgp_mask, bglp, y, key_, info_


def collate_fn_v2(data_batch):
    '''
    for the transformer model implemented by torch
    :param data_batch:
    :return:
    '''
    gl_, gp_, glp_, y_, key_, info_ = map(list, zip(*data_batch))
    bgl = dgl.batch(gl_)
    bgp = dgl.batch(gp_)
    bglp = dgl.batch(glp_)
    y = torch.unsqueeze(torch.stack(y_, dim=0), dim=-1)
    return bgl, bgp, bglp, y, key_, info_

