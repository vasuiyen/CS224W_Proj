# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 06:48:49 2021
"""
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch.cuda.amp import custom_bwd, custom_fwd

from utils import *
from functions import *

from torch_sparse import SparseTensor, matmul


class ImplicitGraph(nn.Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, aggr, kappa=0.99):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa 
        self.aggr = aggr		

        self.W = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = nn.Parameter(torch.FloatTensor(self.m, 1))
		
        self.attn_layer = None
        if self.aggr == "attn":
           #self.attn_layer = nn.Sequential(nn.Linear(out_features, out_features, bias=True), nn.Tanh(), nn.Linear(out_features, 1, bias=False))
           self.attn_layer = nn.Tanh()	   
		   
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300):
        
        self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        
        b_Omega = support_1			
        return ImplicitFunction.apply(self.W, X_0, A, b_Omega, phi, self.aggr, self.attn_layer, fw_mitr, bw_mitr)
