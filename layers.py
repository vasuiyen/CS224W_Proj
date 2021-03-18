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
        
        self.log.debug(f"Layer node channels = {node_channels}")
        
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
            x = self.propagate(edge_index, x=x, u=u)
            
            err = torch.norm(x_old - x, np.inf)
            if err < self.tol:
                break
            if it == self.max_iters - 1:
                self.log.debug(f"Didn't converge: {err}")
            x_old = x            
        return x
    
    def message_and_aggregate(self, edge_index, x):
        return matmul(edge_index, x, reduce = "sum")
    
    def update(self, aggr_out, u):
        x = aggr_out + self.phi(u)
        return self.activation_fn(x)
