# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2023-12-17 23:38:39
# @Last Modified by:   Rexmo
# @Last Modified time: 2024-04-09 13:13:29
import os
from rdkit import Chem
import pickle

path = '/home/jingxuan/TCR/data/our_data/pmhc_seq'
path2 = '/home/jingxuan/TCR/data/our_data/pmhc_struc'
system_idxs = os.listdir(path)
for system_idx in system_idxs:
    file_dir = path+'/'+system_idx+'/'+system_idx+'/ranked_0.pdb'
    tar_dir = path2+'/'+system_idx+'/pmhc.pdb'
    if os.path.exists(file_dir):
        os.system('mkdir -p %s'%(path2+'/'+system_idx))
        os.system('cp %s %s'%(file_dir,tar_dir))
        cmdline = 'cat %s | grep -n TER' % tar_dir
        res = os.popen(cmdline).read()
        ter_idx = int(res.split(':')[0])
        cmdline = 'tail -n +%s %s > %s/pep.pdb' % (ter_idx + 1, tar_dir, path2+'/'+system_idx)
        os.system(cmdline)

        pmhc = Chem.MolFromPDBFile('%s/pmhc.pdb' % (path2+'/'+system_idx))  # not contain H
        pep = Chem.MolFromPDBFile('%s/pep.pdb' % (path2+'/'+system_idx))  # not contain H
        if pmhc and pep:
            Chem.MolToPDBFile(pmhc, '%s/pmhc.pdb' % (path2+'/'+system_idx))
            Chem.MolToPDBFile(pep, '%s/pep.pdb' % (path2+'/'+system_idx))
            with open('%s/comp'% (path2+'/'+system_idx), 'wb') as f:
                pickle.dump([pep, pmhc], f)
    else:
        print(system_idx)






