# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2023-12-18 15:49:38
# @Last Modified by:   Rexmo
# @Last Modified time: 2024-03-26 20:49:42
import os
import re
import argparse
import pickle
import torch
import time
import numpy as np
import pandas as pd
import dgl
from functools import partial
from itertools import repeat
import multiprocessing as mp
from prody import *
from scipy import sparse
from graph_constructor import *
from utils import *
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

def get_input(file_path,start=None,end=None):
    df = pd.read_csv(file_path)
    b_seqs = df['CDR3'].unique()
    return b_seqs


def get_numbering(seq):
    """
    get the IMGT numbering of CDR3 with ANARCI tool
    """
    template = ['GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGTTDQGEVPNGYNVSRSTIEDFPLRLLSAAPSQTSVYFC', 'FGEGSRLTVL']
    # # save fake tcr file

    save_path = '/home/jingxuan/tmp/tmp_%s.fasta'%(seq)

    with open(save_path, 'w+') as f:
        f.write('>0'+'\n')
        total_seq = ''.join([template[0], seq ,template[1]])
        f.write(str(total_seq))
        f.write('\n')
    
    # # using ANARCI to get numbering file
    cmd = ("ANARCI"
            " -i /home/jingxuan/tmp/tmp_%s.fasta  -o /home/jingxuan/tmp/tmp_align_%s --csv"%(seq, seq))
    res = os.system(cmd)
    
    # # parse numbered seqs data
    try:
        df = pd.read_csv('/home/jingxuan/tmp/tmp_align_%s_B.csv'%(seq))
    except FileNotFoundError:
        raise FileNotFoundError('Error: ANARCI failed to align, please check whether ANARCI exists in your environment')
        
    cols = ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117']
    seqs_al = []
    for col in cols:
        if col in df.columns:
            seqs_al_curr = df[col].values
            seqs_al.append(seqs_al_curr)
        else:
            seqs_al_curr = np.full([len(df)], '-')
            seqs_al.append(seqs_al_curr)
    # print(seqs_al)
    seqs_al = [''.join(seq) for seq in np.array(seqs_al).T]
    # print(seqs_al)
    ## merge
    os.remove('/home/jingxuan/tmp/tmp_align_%s_B.csv'%(seq))
    os.remove('/home/jingxuan/tmp/tmp_%s.fasta'%(seq))
    return seqs_al


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

def cdr_align(esm_model, esm_batch_converter, device,
                            b_seq,
                            use_cpu=False,
                            b_max=18):
    status = True
    b_num = len(b_seq)
    if b_num > b_max:
        print(b_seq)
    # construct cdrb graph (gb)

    gb = dgl.DGLGraph().to(device)
    gb.add_nodes(b_max)
    b_embed = cdr_sequence_featurizer_esm(esm_model, esm_batch_converter, b_seq, 
                                                      device)
    align_b = get_numbering(b_seq)[0]
    align_idx = 0
    for idx,residue in enumerate(align_b):
        if residue == '-':
            gb.nodes[idx].data['x'] = torch.zeros(1280).unsqueeze(0).to(device)
            gb.nodes[idx].data['pad'] = torch.zeros(1).to(device)
        else:
            gb.nodes[idx].data['x'] = b_embed[align_idx,:].unsqueeze(0).to(device)
            gb.nodes[idx].data['pad'] = torch.ones(1).to(device)
            align_idx += 1

    if torch.any(torch.isnan(gb.ndata['x'])):
        status = False
        print('nan error', b_seq)
    if status:
        if use_cpu:
            return {'gb': gb.to('cpu')}
        else:
            return {'gb': gb}



    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_dir', type=str, default='/home/jingxuan/TCR/data/all_process.csv', help="which dataset for generation")
    argparser.add_argument('--b_max', type=int, default='18', help="CDR3 beta max length")
    argparser.add_argument('--cdr_dir', type=str, default='/home/jingxuan/TCR/baseline/trap/data_cdr/esm_align', help="cdr_dir")

    args = argparser.parse_args()
    data_dir = args.data_dir
    path_marker = '/'
    home_path = args.cdr_dir
    b_max = args.b_max
    os.system('mkdir -p %s'%home_path)

    
    device = 0
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    # esm
    esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", 
                                             "esm2_t33_650M_UR50D")
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(dev)
    esm_model.eval()

    with torch.no_grad():
        b_seqs = get_input(data_dir)
        for b_seq in b_seqs:
            dic = cdr_align(esm_model, esm_batch_converter, dev,
                                                b_seq,
                                                use_cpu=True, 
                                                b_max=b_max)
            with open(home_path + path_marker + b_seq, 'wb') as f:
                pickle.dump(dic, f)


    