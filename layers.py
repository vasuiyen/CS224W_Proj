# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 06:48:49 2021
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    
    Reference: https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

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
        
    def reset_parameters(self):
        """
        Use Kaiming initialization.
        Greatly improves convergence for deep nets using ReLU.
        A recurrent neural net is infinitely deep. 
        Source: https://pouannes.github.io/blog/initialization/
        """
        stdv = 1. / math.sqrt(self.W.weight.shape[1])
        self.W.weight.data.uniform_(-stdv, stdv)
        self.phi.weight.data.uniform_(-stdv, stdv)
        if self.node_feature_bias: nn.init.uniform_(self.phi.bias)
        if self.node_embedding_bias: nn.init.uniform_(self.W.bias)
        
    def clamp(self, min, max):
        return DifferentiableClamp.apply(self.W.weight, min, max)
    
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
        x = self.W(x)
        x = self.propagate(edge_index, x=(x,x))
        x = x + self.phi(u)
        x = self.activation_fn(x)
        return x
    
    def message(self, x_j):
        """
        Get the message that neighbouring nodes pass to this node. 
        
        @param x_j: 
            Hidden representation of neighbouring nodes.
        """
        return x_j

    def aggregate(self, inputs, index, dim_size = None):
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, 
                             dim_size=dim_size, reduce="sum")
