# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2023-12-17 23:38:09
# @Last Modified by:   Rexmo
# @Last Modified time: 2024-03-26 21:10:40
import pickle
import argparse
import os
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from prody import *
from graph_constructor import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dgl.data.utils import Subset
import torch
from ase import Atoms
from scipy.spatial import distance_matrix
from dscribe.descriptors import ACSF


import warnings
warnings.filterwarnings("ignore")

# Setting up the ACSF descriptor
acsf = ACSF(
    species=['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    r_cut=12.0,
    g2_params=[[4.0, 3.17]],
    g4_params=[[0.1, 3.14, 1]],
)


def epi_featurizer_esm(epi, pmhc, inter_dict1, inter_dict2, inter_dict3, pmhc_atoms_aves,
                              esm_model, esm_batch_converter, sequence, device):
    """
    :param pocket: pdb object generated using parsePDB function
    :param pocket_aves:
    :return:
    """

    # aggregate the ave atom environment descriptor, radius of gyration and BLOSUM62 info for each residual
    gyrations = []
    res_aves = []
    res_aves2 = []
    res_aves3 = []
    
    for idx, residue in enumerate(epi.iterResidues()):
        # radius of gyration
        gyrations.append(calcGyradius(residue))
        inter_atom_ls = []
        atom_num_count = 0
        for i, _ in enumerate(pmhc.iterResidues()):
            num_of_residue = _.numAtoms()
            if i in inter_dict1[idx]:
                inter_atom_ls.extend(pmhc_atoms_aves[atom_num_count:atom_num_count+num_of_residue])
            atom_num_count += num_of_residue
        res_aves.append(torch.mean(torch.stack(inter_atom_ls), axis=0).numpy())  
        
        inter_atom_ls2 = []
        atom_num_count2 = 0
        for i, _ in enumerate(pmhc.iterResidues()):
            num_of_residue = _.numAtoms()
            if i in inter_dict2[idx]:
                inter_atom_ls2.extend(pmhc_atoms_aves[atom_num_count2:atom_num_count2+num_of_residue])
            atom_num_count2 += num_of_residue
        res_aves2.append(torch.mean(torch.stack(inter_atom_ls2), axis=0).numpy())  
        
        inter_atom_ls3 = []
        atom_num_count3 = 0
        for i, _ in enumerate(pmhc.iterResidues()):
            num_of_residue = _.numAtoms()
            if i in inter_dict3[idx]:
                inter_atom_ls3.extend(pmhc_atoms_aves[atom_num_count3:atom_num_count3+num_of_residue])
            atom_num_count3 += num_of_residue
        res_aves3.append(torch.mean(torch.stack(inter_atom_ls3), axis=0).numpy())  
    
    
    gyrations_th = torch.unsqueeze(torch.tensor(gyrations, dtype=torch.float), dim=1)  # len = 1
    esm_info_th = torch.tensor(epi_featurizer_esm(esm_model, esm_batch_converter, sequence, device)['x'], dtype=torch.float)  # len = 1280
    # print(res_aves)
    res_aves_th = torch.tensor(res_aves, dtype=torch.float)  # len = 63
    res_aves_th2 = torch.tensor(res_aves2, dtype=torch.float)  # len = 63
    res_aves_th3 = torch.tensor(res_aves3, dtype=torch.float)  # len = 63
    # print(res_aves_th)
    # print(res_aves_th2)
    # print(res_aves_th3)
    # print(gyrations_th.shape, blosum_info_th.shape, res_aves_th.shape)
    ndata = torch.cat([gyrations_th, res_aves_th, res_aves_th2, res_aves_th3, esm_info_th], dim=-1)
    # print(ndata.shape)
    return {'x': ndata}








def epi_feature_esm(system_path, esm_model, esm_batch_converter, device='cpu'):
    with open(system_path+'/comp', 'rb') as f:
        mol1, mol2 = pickle.load(f)  # mol1 is the ligand, mol2 is the pocket
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()
    # num_atoms = num_atoms_m1 + num_atoms_m2  # total number of atoms
    epi = parsePDB(system_path+'/pep.pdb')
    pmhc = parsePDB(system_path+'/pmhc.pdb')
    epi_num_residues = epi.numResidues()
    pmhc_num_residues = pmhc.numResidues()
    # DScribe package
    atom_ls = []
    atom_ls.extend([atom.GetSymbol() for atom in mol2.GetAtoms()])
    atom_positions = mol2.GetConformer().GetPositions()
    mol_ase = Atoms(symbols=atom_ls, positions=atom_positions)
    res = acsf.create(mol_ase) 
    res_torch = torch.tensor(res, dtype=torch.float)
    # if torch.any(torch.isnan(res_th)):
    #     print(key)
    #     status = False
    epi_atoms_aves = res_torch[:num_atoms_m1].float()
    pmhc_atoms_aves = res_torch[-num_atoms_m2:].float()

    epi_res_geo_center=get_residue_center(epi)
    pmhc_res_geo_center=get_residue_center(pmhc)

    
    dis_matrix = distance_matrix(epi_res_geo_center, pmhc_res_geo_center)
    node_idx = np.where(dis_matrix < dis_threshold1)
    inter_dict = get_interact_dict(node_idx)
    node_idx2 = np.where(dis_matrix < dis_threshold2)
    inter_dict2 = get_interact_dict(node_idx2)
    node_idx3 = np.where(dis_matrix < dis_threshold3)
    inter_dict3 = get_interact_dict(node_idx3)

    epi_seq = struc2seq(epi)
    epi_num = len(epi_seq)
    gp = construct_graph_from_sequence(epi_seq, device)
    gp.ndata.update(epi_featurizer_esm(epi, pmhc, inter_dict, inter_dict2, inter_dict3, pmhc_atoms_aves,
                                              esm_model, esm_batch_converter, epi_seq, device))
    gp.ndata.update({'pad': torch.ones(epi_num).to(device)})

    # add padding node in gp
    gp.add_nodes(epi_max - epi_num)

    # padding
    padding_idx = list(range(epi_num, epi_max))
    gp.nodes[padding_idx].data['x'] = torch.zeros((epi_max - epi_num, 1470)).to(device)
    gp.nodes[padding_idx].data['pad'] = torch.zeros(epi_max - epi_num).to(device)

    return {'gp': gp}


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--struc_path', type=str, default='/home/jingxuan/TCR/data/pmhc_struc')
    argparser.add_argument('--epi_max', type=int, default='12')
    argparser.add_argument('--file_path', type=str, default='/home/jingxuan/TCR/baseline/trap/data_epi')

    args = argparser.parse_args()
    
    dis_threshold1 = 5
    dis_threshold2 = 8
    dis_threshold3 = 15
    epi_max = args.epi_max
    struc_path = args.struc_path
    file_path = args.file_path
    os.system('mkdir -p %s'%file_path)
        
    device = 0
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")
    esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", 
                                             "esm2_t33_650M_UR50D")
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(dev)
    esm_model.eval()
    with torch.no_grad():
        for system_idx in os.listdir(struc_path):
            system_path = struc_path+'/'+system_idx
            epi_feat = epi_feature_esm(system_path, esm_model, esm_batch_converter, dev)
            with open(file_path+'/'+system_idx,'wb') as fo:
                pickle.dump(epi_feat, fo)
