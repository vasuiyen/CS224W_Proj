import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy.sparse import coo_matrix

import numpy as np

from layers import *
from utils import *

class GCN(torch.nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                args, 
                log, 
                **kwargs):

        # TODO: Implement this function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        self.loss = F.nll_loss

        self.log = log

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. More information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## More information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim, args.hidden_dim)] + 
            [GCNConv(args.hidden_dim, args.hidden_dim) for _ in range (args.num_layers - 2) ] + 
            [GCNConv(args.hidden_dim, output_dim)]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(args.hidden_dim) for _ in range (args.num_layers - 1)] 
        )

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        #########################################

        # Probability of an element to be zeroed
        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # TODO: Implement this function that takes the feature tensor x,
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        x, adj_t = data.x, data.edge_index

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as showing in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## More information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)
        for i in range(len(self.bns)):
            
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.convs[-1](x, adj_t)

        out = self.softmax(out)
        
        return out
        #########################################

        return out

class ImplicitGraphNeuralNet(torch.nn.Module):
    """ 
    Recurrent graph neural net model. 
    
    Idea: 
        A feedforward GNN has k layers limiting aggregation to k hops away. 
        Recurrent GNN has infinite layers that share the same parameters.
        This enables aggregation from any distance away. 
        Hidden state is computed based on node features and previous hidden state. 
        
    Details:
        When training the model, initialize a random embedding for each node
        at the start. As the model is trained, the embedding will converge to
        a good embedding. 
        
        Includes softmax to be consistent with GCN implemented above
    
    Implemented based on Gu (2017), equation 1 https://arxiv.org/abs/2009.06211
    """
    def __init__(self,
                input_dim, 
                output_dim, 
                args, 
                log,
                **kwargs):
        """        
        @param input_dim: 
            Node feature dimension
        @param output_dim: 
            Dimension of prediction output
            
        debug: if True, do some debug logging
        """       
        super(ImplicitGraphNeuralNet, self).__init__()
        

        self.node_channels = input_dim
        self.hidden_channels = args.hidden_dim
        self.tol = args.tol
        self.kappa = args.kappa
        self.max_iters = args.max_forward_iterations
        self.drop_prob = args.drop_prob
        num_nodes = kwargs.pop('num_nodes')

        self.spectral_radius_dict = {}

        # Initialize the neural net
        self.graph_layer = GeneralGraphLayer(
            in_channels = args.hidden_dim, 
            out_channels = args.hidden_dim, 
            node_channels = input_dim, 
            **kwargs
        )
        self.prediction_head = nn.Linear(args.hidden_dim, output_dim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.log = log
        self.log.debug(f"Initializing IGNN with node_channels={input_dim}, hidden_channels={args.hidden_dim}, prediction_channels={output_dim}")
        
    def reset_parameters(self):
        self.graph_layer.reset_parameters()
        self.prediction_head.reset_parameters()

    def project_recurrent_weight(self, spectral_radius):
        projection_norm_inf(self.graph_layer.W.weight, self.kappa / spectral_radius)
    
    def forward(self, data):
        """        
        @param data: Graph object
            
        @return y: Model outputs after convergence.
        """
        node_index, node_feature, edge_index, adj_matrix = data.orig_node_idx, data.x, data.edge_index, data.adj_matrix
        num_nodes = node_feature.shape[0]

        if hasattr(data, 'batch_index'):
            if data.batch_index not in self.spectral_radius_dict:
                self.spectral_radius_dict[data.batch_index] = compute_spectral_radius(adj_matrix)

            spectral_radius = self.spectral_radius_dict.get(data.batch_index)
                
        else:
            spectral_radius = compute_spectral_radius(adj_matrix)
        
        self.project_recurrent_weight(spectral_radius)

        # Use zeros to initialize embeddings
        x = torch.zeros((num_nodes, self.hidden_channels)).to(node_feature.device)
        
        # Train embeddings to convergence; this constitutes 1 forward pass
        for it in range(self.max_iters):
            x_new = self.graph_layer(x, node_feature, edge_index)
            err = torch.norm(x_new - x, np.inf)
            if torch.norm(x_new - x, np.inf) < self.tol:
                break
            x = x_new
            if it == self.max_iters - 1:
                self.log.info(f"Didn't converge: {err}")

        x = F.dropout(x, self.drop_prob, training=self.training)

        out = self.prediction_head(x)
        return self.softmax(out)

class DataParallelWrapper(torch.nn.DataParallel):  
    """ torch.nn.DataParallel that supports clamp() and reset_parameters()"""     
    def reset_parameters(self):
        self.module.reset_parameters()
