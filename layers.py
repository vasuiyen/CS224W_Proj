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

    def __init__(self, in_features, out_features, num_node, kappa=0.99, init_type="uniform"):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa     

        self.W = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = nn.Parameter(torch.FloatTensor(self.m, 1))
        self.init(init_type)

    def init(self, init_type):

        if init_type == 'uniform':
            stdv = 1. / math.sqrt(self.W.size(1))
            self.W.data.uniform_(-stdv, stdv)
            self.Omega_1.data.uniform_(-stdv, stdv)
            self.Omega_2.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

        elif init_type == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.W.data)
            torch.nn.init.kaiming_uniform_(self.Omega_1.data)
            torch.nn.init.kaiming_uniform_(self.Omega_2.data)
            torch.nn.init.kaiming_uniform_(self.bias.data)

        elif init_type == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.W.data)
            torch.nn.init.kaiming_normal_(self.Omega_1.data)
            torch.nn.init.kaiming_normal_(self.Omega_2.data)
            torch.nn.init.kaiming_normal_(self.bias.data)


    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300):
        
        self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        
        b_Omega = support_1
        return ImplicitFunction.apply(self.W, X_0, A, b_Omega, phi, fw_mitr, bw_mitr)

