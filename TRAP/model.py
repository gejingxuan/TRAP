# -*- coding: utf-8 -*-
# @Author: Rexmo
# @Date:   2024-01-19 15:25:43
# @Last Modified by:   Rexmo
# @Last Modified time: 2024-03-26 21:18:18
import torch as th
import numpy as np
from torch.nn import LayerNorm
import torch.nn.init as INIT
import copy
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl
import math
from torch.nn import Linear
from .transformer_torch import *

torch.set_default_tensor_type('torch.FloatTensor')



class CrossAttnEncoder(nn.Module):
    def __init__(self, in_feat_b, in_feat_p, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout):
        '''
        :param in_feat_gp: the length of input features of pocket node
        :param d_model: Embedding Size
        :param d_ff: FeedForward dimension
        :param d_k: dimension of K(=Q)
        :param d_v: dimension of  V
        :param n_heads:  number of heads in Multi-Head Attention
        :param n_layers: number of CrossAttnEncoder
        :param dropout: dropout ratio
        '''
        super(CrossAttnEncoder, self).__init__()
        self.in_feat_b = in_feat_b
        self.in_feat_p = in_feat_p
        
        self.src_emb_b = nn.Linear(in_feat_b, d_model)
        self.src_emb_p = nn.Linear(in_feat_p, d_model)
        
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([CrossAttnLayer(d_model, d_ff, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])
    def forward(self, gb, gp):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # cdrb: from cdrb graph features to embedding
        cdrb_ndata_x = gb.ndata['x'].view(gb.batch_size, -1, self.in_feat_b)  # [batch_size, src_len, in_feat_b]
        cdrb_ndata_pad = gb.ndata['pad'].view(gb.batch_size, -1)  # [batch_size, src_len]

        cdrb_enc_outputs = self.src_emb_b(cdrb_ndata_x)  # [batch_size, src_len, d_model]
        cdrb_enc_outputs = self.pos_emb(cdrb_enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        
        # epitope: from epitope graph features to embedding
        epi_ndata_x = gp.ndata['x'].view(gp.batch_size, -1, self.in_feat_p)  # [batch_size, src_len, in_feat_p]
        epi_ndata_pad = gp.ndata['pad'].view(gp.batch_size, -1)  # [batch_size, src_len]

        epi_enc_outputs = self.src_emb_p(epi_ndata_x)  # [batch_size, src_len, d_model]
        epi_enc_outputs = self.pos_emb(epi_enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        
        # enc_attn_mask = get_attn_pad_mask(cdrb_ndata_pad, epi_ndata_pad)  # [batch_size, src_len, src_len]
        enc_attn_mask = get_attn_pad_mask(epi_ndata_pad, cdrb_ndata_pad)
        enc_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(epi_enc_outputs, cdrb_enc_outputs, cdrb_enc_outputs, enc_attn_mask)
            enc_attns.append(enc_self_attn)
        return enc_outputs, enc_attns



class Encoder_(nn.Module):
    def __init__(self, in_feat, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout):
        '''
        :param in_feat_gp: the length of input features of pocket node
        :param d_model: Embedding Size
        :param d_ff: FeedForward dimension
        :param d_k: dimension of K(=Q)
        :param d_v: dimension of  V
        :param n_heads:  number of heads in Multi-Head Attention
        :param n_layers: number of Encoder  Layer
        :param dropout: dropout ratio
        '''
        super(Encoder_, self).__init__()
        self.in_feat = in_feat
        self.src_emb = nn.Linear(in_feat, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, bg):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        ndata_x = bg.ndata['x'].view(bg.batch_size, -1, self.in_feat)  # [batch_size, src_len, in_feat]
        ndata_pad = bg.ndata['pad'].view(bg.batch_size, -1)  # [batch_size, src_len]

        enc_outputs = self.src_emb(ndata_x)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(ndata_pad, ndata_pad)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h
    

class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, dropout):
        '''
        :param d_model: Embedding Size
        :param d_ff: FeedForward dimension
        :param d_k: dimension of K(=Q)
        :param d_v: dimension of V
        :param n_heads: number of heads in Multi-Head Attention
        '''
        super(CrossAttnLayer, self).__init__()

        self.enc_MultiHead_Attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_input_Q, enc_inputs_K, enc_inputs_V, enc_attn_mask):
        '''
        enc_input_Q/K/V: [batch_size, src_len, d_model]
        enc_MultiHead_Attn: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_MultiHead_Attn(enc_input_Q, enc_inputs_K, enc_inputs_V, enc_attn_mask) # [Q, (K, V)] same K and V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn