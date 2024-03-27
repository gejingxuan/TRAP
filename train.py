# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2023-12-19 14:20:29
# @Last Modified by:   Rexmo
# @Last Modified time: 2024-03-26 21:15:33
import torch
from torch import nn
import torch.nn.functional as F
from model import Encoder_, FC, CrossAttnEncoder
from torch.utils.data import DataLoader,Subset
import pandas as pd
import numpy as np
import warnings
from graph_constructor import *
from prefetch_generator import BackgroundGenerator
from utils import *
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
import argparse
import datetime

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_input(file_path):
    df = pd.read_csv(file_path)
    b_seqs = df['CDR3'].values
    p_seqs = df['pmhc'].values
    labels = df['labels'].values
    keys = df['idx'].values
    return b_seqs, p_seqs, labels, keys

def cal_aupr(data_true,data_pred):
    precision, recall, _ = precision_recall_curve(data_true,data_pred)
    aupr = auc(recall, precision)

    return aupr

def get_feat(cdr_dir, epi_dir, label, key, b_seq, p_seq):
    
    with open(cdr_dir+'/'+b_seq,'rb') as f:
        gb_dic = pickle.load(f)
        gb = gb_dic['gb']
    with open(epi_dir+'/'+p_seq,'rb') as f:
        gp_dic = pickle.load(f)
        gp = gp_dic['gp']
    
    return {'gb': gb, 'gp': gp, 'label': label, 'key': key}
    



def run_clip_train_epoch(model, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bgb, bgp, y, key_  = batch
        bgb, bgp, y = bgb.to(device), bgp.to(device), y.to(device)
        _, loss_clip = model(bgb, bgp)
        loss_clip.backward()
        optimizer.step()

def run_bi_train_epoch(model, train_dataloader, loss_fn, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bgb, bgp, y, key_ = batch
        bgb, bgp, y = bgb.to(device), bgp.to(device), y.to(device)
        pred, _ = model(bgb, bgp)
        # print(pred.size())
        # print(y.size())
        loss_bi = loss_fn(pred, y)
        # print(loss_bi)
        loss_bi.backward()
        optimizer.step()

def run_a_eval_epoch(model, validation_dataloader, loss_fn, device):
    true = []
    pred = []
    key = []
    loss1 = []
    loss2 = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # TRAPModel.zero_grad()
            bgb, bgp, y, key_ = batch
            bgb, bgp, y = bgb.to(device), bgp.to(device), y.to(device)
            pred_, loss_= model(bgb, bgp)
            loss_bi = loss_fn(pred_, y)
            pred.extend(pred_.squeeze().data.cpu().numpy())
            true.extend(y.squeeze().data.cpu().numpy())
            key.extend(key_)
            loss1.append(loss_bi.to('cpu'))
            loss2.append(loss_.to('cpu'))
            # print(np.array(loss).mean())
    return true, pred, key, np.array(loss1).mean(), np.array(loss2).mean()



def collate_fn_v2(data_batch):
    '''
    for the transformer model implemented by torch
    :param data_batch:
    :return:
    '''
    gb_, gp_, y_, key_ = map(list, zip(*data_batch))
    bgb = dgl.batch(gb_)
    bgp = dgl.batch(gp_)
    y = torch.unsqueeze(torch.stack(y_, dim=0), dim=-1)
    return bgb, bgp, y, key_

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class TrapDataset(object):
    def __init__(self, cdr_dir=None, epi_dir=None, 
                 data_dirs=None, graph_ls_file=None, graph_dic_path=None, path_marker='/', 
                 del_tmp_files=False, p_max=12, b_max=18):
        self.data_dirs = data_dirs
        self.graph_ls_file = graph_ls_file
        self.graph_dic_path = graph_dic_path
        self.path_marker = path_marker
        self.del_tmp_files = del_tmp_files
        self.p_max = p_max
        self.b_max = b_max
        self.cdr_dir = cdr_dir
        self.epi_dir = epi_dir
        self._pre_process()


    def _pre_process(self):
        if os.path.exists(self.graph_ls_file):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_file, 'rb') as f:
                data = pickle.load(f)
            self.gb = data['gb']
            self.gp = data['gp']
            self.labels = data['labels']
            self.keys = data['keys']
        else:
            # mk dic path
            if os.path.exists(self.graph_dic_path):
                pass
            else:
                cmdline = 'mkdir -p %s' % (self.graph_dic_path)
                os.system(cmdline)
                self.b_seqs, self.p_seqs, self.binds, self.keys= get_input(self.data_dirs)
                print('Generate CDRb-epitope graph...')
            
            # memory friendly
                st = time.time()
                print("main process start >>> pid={}".format(os.getpid()))
                for i in range(len(self.keys)):
                    dic = get_feat(self.cdr_dir, self.epi_dir, self.binds[i], self.keys[i],self.b_seqs[i], self.p_seqs[i])
                    with open(self.graph_dic_path + path_marker + str(dic['key']), 'wb') as f:
                        pickle.dump(dic, f)
                print("main process end (time:%s S)\n" % (time.time() - st))

            # collect the generated graph for each complex
            self.gb = []
            self.gp = []
            self.labels = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.gb.append(graph_dic['gb'])
                    self.gp.append(graph_dic['gp'])
                    self.labels.append(graph_dic['label'])
            
            # store to the disk
            with open(self.graph_ls_file, 'wb') as f:
                pickle.dump({'gb': self.gb, 'gp': self.gp, 'labels': self.labels,
                             'keys': self.keys}, f)

            # delete the temporary files
            if self.del_tmp_files:
                cmdline = 'rm -rf %s' % self.graph_dic_path  # graph_dic_path
                os.system(cmdline)
    
    def __getitem__(self, indx):
        return self.gb[indx], self.gp[indx], torch.tensor(float(self.labels[indx])), \
               self.keys[indx]

    def __len__(self):
        return len(self.keys)

class TransTRAP(nn.Module):

    def __init__(self, p_max=12, b_max=18, in_feat_b=1280, in_feat_p=84, d_model=200, d_ff=512, d_k=128, d_v=128, n_heads=4, n_layers=3,
                 dropout=0.20, d_graph_layer=200, batch_size=256):
        super(TransTRAP, self).__init__()
        self.b_max = b_max
        self.p_max = p_max
        self.d_model = d_model
        self.batch_size = batch_size
        self.d_graph_layer = d_graph_layer
        self.temperature = 0.2
        # Multi-Heads encoder layer for sequence
        self.b_encoder = Encoder_(in_feat=in_feat_b, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                n_layers=n_layers, dropout=dropout)
        self.p_encoder = Encoder_(in_feat=in_feat_p, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                n_layers=n_layers, dropout=dropout)
        
        self.b_projection = nn.Linear(d_model*self.b_max, self.d_model, bias=False)
        self.p_projection = nn.Linear(d_model*self.p_max, self.d_model, bias=False)

        self.fc = FC(self.d_model*2, d_FC_layer, n_FC_layer, dropout, n_tasks)


    def forward(self, bgb, bgp):
        # node representation calculation for the ligand residue sequence
        b_encoder, b_attn = self.b_encoder(bgb)  # [batch_size, src_len, d_model]
        # print(b_encoder.size())
        p_encoder, p_attn = self.p_encoder(bgp)
        b_embed = b_encoder.view(-1,self.b_max*self.d_model)
        p_embed = p_encoder.view(-1,self.p_max*self.d_model)

        # print(p_embed.size())
        
        b_encoder = self.b_projection(b_embed)
        p_encoder = self.p_projection(p_embed)
        # print(b_encoder.size()) # [batchsize,200]

        # Calculating the Loss
        b_encoder = b_encoder / b_encoder.norm(p=2, dim=-1, keepdim=True)
        p_encoder = p_encoder / p_encoder.norm(p=2, dim=-1, keepdim=True)
        logits = (b_encoder @ p_encoder.T) / self.temperature

        binding_input = torch.cat((b_encoder,p_encoder),dim=-1)
        binding_onput = self.fc(binding_input)
        loss = clip_loss(logits)
        pred_bi = torch.sigmoid(binding_onput)
        # print(pred_bi.size())
        # print(loss)

        return pred_bi, loss





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # model training parameters
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10**-4, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=500, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=40, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=0.00, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=5, help="the number of independent runs")
    # transformer parameters
    argparser.add_argument('--n_layers', type=int, default=3, help='the number of encoder layer')
    argparser.add_argument('--in_feat_b', type=int, default=1280)
    argparser.add_argument('--in_feat_p', type=int, default=1470)
    argparser.add_argument('--d_ff', type=int, default=512)
    argparser.add_argument('--d_k', type=int, default=128)
    argparser.add_argument('--d_v', type=int, default=128)
    argparser.add_argument('--n_heads', type=int, default=4)
    argparser.add_argument('--d_model', type=int, default=200)
    argparser.add_argument('--dropout', type=float, default=0.3)
    # glp gnn parameters
    argparser.add_argument('--d_graph_layer', type=int, default=200, help='the output dim of cdrb-epitope layers')
    # MLP predictor parameters
    argparser.add_argument('--d_FC_layer', type=int, default=100, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--n_tasks', type=int, default=1)
    # others
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--num_process', type=int, default=1,
                           help='number of process for generating graphs')
    argparser.add_argument('--dic_path_suffix', type=str, default='1')
    argparser.add_argument('--b_max', type=int, default=18,
                           help='used for padding, the maximum number of '
                                'cdr3b residue in the dataset')
    argparser.add_argument('--p_max', type=int, default=12,
                           help='used for padding, the maximum number of '
                                'epitope residues in the dataset')
    argparser.add_argument('--test_scripts', type=int, default=0,
                           help='whether to test the scripts can run successfully '
                                'using part of datasets (1 for True, 0 for False)')
    argparser.add_argument('--home_path', type=str, default='/home/jingxuan/TCR/baseline/TRAP/data_randomly',
                           help='path to run the scripts and store the results ')
    argparser.add_argument('--cdr_dir', type=str, default='/home/jingxuan/TCR/baseline/TRAP/data_cdr')
    argparser.add_argument('--epi_dir', type=str, default='/home/jingxuan/TCR/baseline/TRAP/data_epi')
    argparser.add_argument('--data_dir', type=str, default='/home/jingxuan/TCR/data/our_data/data_v1/randomly')

    args = argparser.parse_args()

    # print(args)
    # model training parameters
    gpuid, lr, epochs, batch_size, tolerance, patience, l2, repetitions = args.gpuid, args.lr, args.epochs, args.batch_size, args.tolerance, args.patience, \
                                                                          args.l2, args.repetitions
    # gp transformer parameters
    in_feat_b, in_feat_p, d_ff, d_k, d_v, n_heads, n_layers, d_model, dropout = args.in_feat_b, args.in_feat_p, args.d_ff, args.d_k, args.d_v, args.n_heads, args.n_layers, \
                                                                      args.d_model, args.dropout
    # glp gnn parameters
    d_graph_layer = args.d_graph_layer
    # MLP predictor parameters
    d_FC_layer, n_FC_layer, n_tasks = args.d_FC_layer, args.n_FC_layer, args.n_tasks
    # other parameters
    home_path = args.home_path
    cdr_dir = args.cdr_dir
    epi_dir = args.epi_dir
    data_dir = args.data_dir
    p_max = args.p_max
    b_max = args.b_max


    # print(torch.cuda.is_available())
    if args.test_scripts == 1:
        epochs = 5
        repetitions = 2
        batch_size = 8
        limit = 20
    else:
        limit = None
    
    path_marker = '/'
    reuslt_path = '%s/result' %home_path
    os.system('mkdir -p %s/result' %home_path)
    os.system('mkdir -p %s/model_save' %reuslt_path)
    os.system('mkdir -p %s/stats' %reuslt_path)


    positive_dataset = TrapDataset(cdr_dir= cdr_dir, epi_dir= epi_dir, data_dirs=data_dir+'/positive.csv',
                            graph_ls_file=home_path + path_marker +'positive.bin',
                            graph_dic_path=home_path + path_marker + 'positive', path_marker='/',
                            del_tmp_files=True, p_max=p_max, b_max=b_max)
    
    positive_idx = np.arange(0, len(positive_dataset))

    kfold_dataset = TrapDataset(cdr_dir= cdr_dir, epi_dir= epi_dir, data_dirs=data_dir+'/train.csv',
                            graph_ls_file=home_path + path_marker + 'train.bin',
                            graph_dic_path=home_path + path_marker + 'train', path_marker='/',
                            del_tmp_files=True, p_max=p_max, b_max=b_max)
    
    test_dataset = TrapDataset(cdr_dir= cdr_dir, epi_dir= epi_dir, data_dirs=data_dir+'/test.csv',
                            graph_ls_file=home_path + path_marker + 'test.bin',
                            graph_dic_path=home_path + path_marker + 'test', path_marker='/',
                            del_tmp_files=True, p_max=p_max, b_max=b_max)


    kfold_dataloader = DataLoaderX(kfold_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate_fn_v2)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                      collate_fn=collate_fn_v2)
    
    stat_res = []

    for repetition_th in range(repetitions):
        torch.cuda.empty_cache()
        dt = datetime.datetime.now()
        filename = reuslt_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        # print('Independent run %s' % repetition_th)
        # print('model file %s' % filename)
        set_random_seed(repetition_th*6)
        
        # model
        TRAPModel = TransTRAP(p_max=p_max, b_max=b_max, in_feat_b=in_feat_b, in_feat_p=in_feat_p, 
                             d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                          n_layers=n_layers, dropout=dropout, d_graph_layer=d_graph_layer,
                          batch_size=batch_size)
        print('number of parameters : ', sum(p.numel() for p in TRAPModel.parameters() if p.requires_grad))
        # if repetition_th == 0:
            # print(TRAPModel)
        
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
        TRAPModel.to(device)
        optimizer = torch.optim.Adam(TRAPModel.parameters(), lr=lr, weight_decay=l2)
        loss_fn = FocalLoss(gamma=2, alpha=30 / (30 + 1))

        
        


        kf = KFold(n_splits=5)  # 5-fold

        for train_index, valid_index in kf.split(positive_idx):
            stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=filename)
            # print(valid_index)
            train_dataset = Subset(positive_dataset, train_index)
            valid_dataset = Subset(positive_dataset, valid_index)

            train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers,
                                        collate_fn=collate_fn_v2)
            valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=collate_fn_v2)
        
            train_loss_record = []
            train_auc_record = []
            valid_loss_record = []
            valid_auc_record = []
            for epoch in range(epochs):
                st = time.time()
                # train
                run_clip_train_epoch(TRAPModel, train_dataloader, optimizer, device)
                run_bi_train_epoch(TRAPModel, kfold_dataloader, loss_fn, optimizer, device)

                # validation
                train_true,  train_pred, train_key, train_loss, train_loss2 = run_a_eval_epoch(TRAPModel, train_dataloader, loss_fn, device)
                valid_true,  valid_pred, valid_key, valid_loss, valid_loss2 = run_a_eval_epoch(TRAPModel, valid_dataloader, loss_fn, device)
                kfold_true,  kfold_pred, kfold_key, kfold_loss, kfold_loss2 = run_a_eval_epoch(TRAPModel, kfold_dataloader, loss_fn, device)
                # print(train_loss)
                # print(valid_loss)
                # train_auc = roc_auc_score(train_true, train_pred)
                kfold_auc = roc_auc_score(kfold_true, kfold_pred)
                early_stop = stopper.step(valid_loss, TRAPModel)

                end = time.time()

                if early_stop:
                    break
                # print("epoch:%s \t train_loss:%.4f \t valid_loss:%.4f \t time:%.3f s" %(
                #         epoch, train_loss, valid_loss, end - st))
                print("epoch:%s\ttrain_loss:%.4f\ttrain_loss2:%.4f\tvalid_loss:%.4f\tvalid_loss2:%.4f\tkfold_loss:%.4f\tkfold_loss2:%.4f\tkfold_auc:%.4f\ttime:%.3fs" %(
                        epoch, train_loss, train_loss2, valid_loss, valid_loss2, kfold_loss, kfold_loss2, kfold_auc, end - st))

        # load the best model
        stopper.load_checkpoint(TRAPModel)



        
        kfold_true, kfold_pred, kfold_keys, kfold_attns, _ = run_a_eval_epoch(TRAPModel, kfold_dataloader,
                                                                                        loss_fn, device)
        test_true, test_pred, te_keys, test_attns, _ = run_a_eval_epoch(TRAPModel, test_dataloader,
                                                                                        loss_fn, device)


        pd_kfold = pd.DataFrame({'key': kfold_keys, 'kfold_true': kfold_true, 'kfold_pred': kfold_pred})
        pd_te = pd.DataFrame({'key': te_keys, 'test_true': test_true, 'test_pred': test_pred})

        pd_kfold.to_csv(
            reuslt_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_kfold.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_te.to_csv(
            reuslt_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)

        kfold_loss = loss_fn(torch.tensor(kfold_pred, dtype=torch.float),
                             torch.tensor(kfold_true, dtype=torch.float))
        test_loss = loss_fn(torch.tensor(test_pred, dtype=torch.float), 
                             torch.tensor(test_true, dtype=torch.float))
        
        kfold_auc = roc_auc_score(kfold_true, kfold_pred)
        test_auc = roc_auc_score(test_true, test_pred)
        
        kfold_aupr = cal_aupr(kfold_true, kfold_pred)
        test_aupr = cal_aupr(test_true, test_pred)





        print('***The Best TRAP model***')
        print("epoch:%s kfold_loss:%.4f te_loss:%.4f kfold_auc:%.4f te_auc:%.4f" % (
            epoch, kfold_loss, test_loss, kfold_auc, test_auc))
        
        stat_res.append([repetition_th, 'kfold', kfold_loss, kfold_auc, kfold_aupr])
        stat_res.append([repetition_th, 'test', test_loss, test_auc, test_aupr])

        
    
    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'loss', 'auc', 'aupr'])
    stat_res_pd.to_csv(
        reuslt_path + path_marker + 'stats' + path_marker +  '{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)

    kfold_aupr_mean=stat_res_pd[stat_res_pd.group == 'kfold'].mean().values[-1]
    kfold_aupr_std= stat_res_pd[stat_res_pd.group == 'kfold'].std().values[-1]
    test_aupr_mean=stat_res_pd[stat_res_pd.group == 'test'].mean().values[-1]
    test_aupr_std= stat_res_pd[stat_res_pd.group == 'test'].std().values[-1]
    
    kfold_auc_mean=stat_res_pd[stat_res_pd.group == 'kfold'].mean().values[-2]
    kfold_auc_std= stat_res_pd[stat_res_pd.group == 'kfold'].std().values[-2]
    test_auc_mean=stat_res_pd[stat_res_pd.group == 'test'].mean().values[-2]
    test_auc_std= stat_res_pd[stat_res_pd.group == 'test'].std().values[-2]
    

    
    output = ''
    output += 'TRAP\tkfold Set\t' + str(kfold_auc_mean) + '\t' + str(kfold_auc_std) + '\t' + str(kfold_aupr_mean) + '\t' + str(kfold_aupr_std) +'\n'
    output += 'TRAP\tTest Set\t' + str(test_auc_mean) + '\t' + str(test_auc_std)+ '\t' + str(test_aupr_mean) + '\t' + str(test_aupr_std) +'\n'
    print(output)
    with open(reuslt_path + path_marker + 'stats' + path_marker +  '{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), 'w')as fo:
        fo.write(output)
    
