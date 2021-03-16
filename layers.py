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


class GeneralGraphLayer(MessagePassing):
    """ A general graph layer.  
    Performs:
    1) Propagate messages
    2) Aggregate messages
    3) Update node representation
    
    Implemented based on Gu (2017), equation 1 https://arxiv.org/abs/2009.06211
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 node_channels: int, 
                 activation_fn = F.relu,
                 node_dim = 0,
                 node_feature_bias = True,
                 node_embedding_bias = True,
                 
                 max_iters = 1,
                 tol = 3e-6,
                 log = None,
                 
                 **kwargs):  
        super(GeneralGraphLayer, self).__init__(**kwargs)
        # Node feature weights. Named \phi in paper equation (1)
        self.phi = nn.Linear(node_channels, out_channels)
        # Node embedding weights. Named W in paper equation (1)
        self.W = nn.Linear(in_channels, out_channels)
        # Nonlinearity to apply
        self.activation_fn = activation_fn
        # The dimension corresponding to different nodes. 
        # E.g. if inputs are (num_nodes, node_dim) then node_dim = 0
        # Used in self.aggregate
        self.node_dim = node_dim
        self.node_feature_bias = node_feature_bias
        self.node_embedding_bias = node_embedding_bias
        
        self.log = log
        self.max_iters = max_iters
        self.tol = tol
        
    def reset_parameters(self):
        """
        Use same initialization as original implementation.
        
        TODO: Test Kaiming initialization instead which should be a lot better. 
        Source: https://pouannes.github.io/blog/initialization/
        """
        stdv = 1. / math.sqrt(self.W.weight.shape[1])
        self.W.weight.data.uniform_(-stdv, stdv)
        self.phi.weight.data.uniform_(-stdv, stdv)
        if self.node_feature_bias: nn.init.uniform_(self.phi.bias)
        if self.node_embedding_bias: nn.init.uniform_(self.W.bias)
    
    def forward(self, x, u, edge_index):
        """
        @param x: 
            Hidden node representation at step T.
            Shape: (batch_size, hidden_dim)
        @param u: 
            Base node features. 
            Shape: (batch_size, node_dim)
        @param edge_index: 
            A tensor containing (source, target) node indexes
            Shape: (2, num_edges)
    
        @return: Node representation at step T+1. 
        """
        
        x_old = x
        for it in range(self.max_iters):  
            x = self.W(x)
            x = self.propagate(edge_index, x=(x,x))
            x = x + self.phi(u)
            x = self.activation_fn(x)
            
            err = torch.norm(x_old - x, np.inf)
            if err < self.tol:
                break
            if it == self.max_iters - 1:
                self.log.info(f"Didn't converge: {err}")
            x_old = x            
        return x
    

def message_and_aggregate(edge_index, node_feature_src):
    return matmul(edge_index, node_feature_src, reduce = "sum")


class ImplicitGraph(nn.Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = nn.Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)

