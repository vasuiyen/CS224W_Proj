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
        
        self.args = args

        self.node_channels = input_dim
        self.hidden_channels = args.hidden_dim
        self.kappa = args.kappa
        self.drop_prob = args.drop_prob
        self.spectral_radius_dict = {}
        num_nodes = kwargs.pop('orig_num_nodes')
        
        self.log = log
        self.log.debug(f"Model node channels = {self.node_channels}")
        
        # Initialize the implicit graph layers
        self.implicit_graph_layers = nn.ModuleList([
            ImplicitGraph(input_dim, args.hidden_dim, num_nodes, args.kappa, args.init_type) for l in range(args.num_layers)
        ])
        
        self.prediction_head = nn.Linear(args.hidden_dim, output_dim)
        
        if self.args.embed_type != 'zero':
            self.embedding = nn.Embedding(num_nodes, args.hidden_dim)
            self.embedding.weight.requires_grad = False

        if self.args.embed_type == 'random':
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

    def reset_parameters(self):
        self.prediction_head.reset_parameters()
        if self.args.embed_type == 'persistent':
            self.embedding.reset_parameters()

    
    def forward(self, data):
        """        
        @param data: Graph object
            
        @return y: Model outputs after convergence.
        """
        node_index, node_feature, edge_index, adj_matrix = data.orig_node_idx, data.x, data.edge_index, data.adj_matrix
        adj_t = data.adj_t
        num_nodes = node_feature.shape[0]

        if hasattr(data, 'batch_index'):
            if data.batch_index not in self.spectral_radius_dict:
                self.spectral_radius_dict[data.batch_index] = compute_spectral_radius(adj_matrix)

            spectral_radius = self.spectral_radius_dict.get(data.batch_index)
                
        else:
            spectral_radius = compute_spectral_radius(adj_matrix)

        adj_matrix = torch.sparse.FloatTensor(
            torch.LongTensor(np.vstack((adj_matrix.row, adj_matrix.col))),
            torch.FloatTensor(adj_matrix.data),
            adj_matrix.shape).to(node_feature.device)

        if self.args.embed_type == 'zero':
            x = torch.zeros(num_nodes, self.hidden_channels).to(node_feature.device)
        else:
            x = self.embedding(node_index)
        
        # Train embeddings to convergence; this constitutes 1 forward pass
        self.log.debug(f"Model u feature shape = {node_feature.shape}")

        for implicit_graph in self.implicit_graph_layers:
            x = implicit_graph(torch.transpose(x, 0, 1), adj_matrix, torch.transpose(node_feature, 0, 1), F.relu, spectral_radius, 
            self.args.max_forward_iterations, self.args.max_forward_iterations).T

        if self.args.embed_type == 'persistent' and self.training == True:
            self.embedding.weight[node_index] = x.detach().clone()
        
        x = F.dropout(x, self.drop_prob, training=self.training)

        return self.prediction_head(x)

class DataParallelWrapper(torch.nn.DataParallel):  
    """ torch.nn.DataParallel that supports clamp() and reset_parameters()"""     
    def reset_parameters(self):
        self.module.reset_parameters()
