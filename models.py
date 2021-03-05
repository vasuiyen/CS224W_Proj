import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv

from layers import *

class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
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

class RecurrentGraphNeuralNet(torch.nn.Module):
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
             node_channels: int,
             hidden_channels: int,
             prediction_channels: int,
             num_nodes: int,
             return_embeds: bool = False,
             debug: bool = False,
             **kwargs):
        """        
        @param node_channels: 
            Node feature dimension
        @param hidden_channels: 
            Model hidden dimension
        @param prediction_channels: 
            Dimension of prediction output
        @param num_nodes:
            Number of nodes in the graph
            
        return_embeds: if True, return a tuple of (hidden_state, output)
        debug: if True, do some debug logging
        """       
        super(RecurrentGraphNeuralNet, self).__init__()
        self.graph_layer = GeneralGraphLayer(
            in_channels = hidden_channels, 
            out_channels = hidden_channels, 
            node_channels = node_channels, **kwargs)
        self.prediction_head = nn.Linear(hidden_channels, prediction_channels)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.node_embedding = nn.Embedding(num_nodes, hidden_channels)
        # Manually turn off embedding training
        # We update the embeddings manually
        self.node_embedding.weight.requires_grad = False
        self.debug = debug
        
        if debug:
            print(f"Initializing RGNN with node_channels={node_channels}, hidden_channels={hidden_channels}, prediction_channels={prediction_channels}")
        
    def reset_parameters(self):
        self.graph_layer.reset_parameters()
        self.prediction_head.reset_parameters()
        self.node_embedding.reset_parameters()
        
    def clamp(self, min, max):
        self.graph_layer.clamp(min, max)
    
    def forward(self, node_index, node_feature, edge_index):
        """        
        @param node_index: 
            Indices of the nodes being passed in.
            Shape: (batch_size, 1)
        @param u: 
            Base node features. 
            Shape: (batch_size, node_channels)
        @param edge_index: 
            A tensor containing (source, target) node indexes
            Shape: (2, num_edges)
            
        @return y: Model outputs at step T+1. 
        """
        # testing code
        if self.debug:
            x_orig = self.node_embedding(node_index)            
            print(f"Inputs: node_index shape {node_index.shape}, node_feature shape {node_feature.shape}")
        x = self.node_embedding(node_index)
        x = self.graph_layer(x, node_feature, edge_index)
        if self.debug:
            assert (x_orig - x).abs().sum() > 0
        self.node_embedding.weight[node_index] = x.detach()
        out = self.prediction_head(x)
        return self.softmax(out)
    
class DeepSnapWrapper(torch.nn.Module):
    """
    Wrap a model to accept DeepSnap batches instead of raw tensors. 
    """
    def __init__(self, model): 
        """
        @param model:
            torch.nn.Module
            expected signature: model(node_index, node_feature, edge_index)
        """
        super(DeepSnapWrapper, self).__init__()
        self.model = model 
        
    def reset_parameters(self):
        self.model.reset_parameters()
        
    def clamp(self, min, max):
        self.model.clamp(min, max)
    
    def forward(self, batch):
        """        
        @param batch:
            A DeepSnap.Batch object
        """
        node_index = batch.node_index
        node_feature = batch.node_feature
        edge_index = batch.edge_index
        return self.model(node_index, node_feature, edge_index)
    
    
class DataParallelWrapper(torch.nn.DataParallel):  
    """ torch.nn.DataParallel that supports clamp() and reset_parameters()"""     
    def reset_parameters(self):
        self.module.reset_parameters()
    def clamp(self, min, max):
        try:
            self.module.clamp(min, max)
        except:
            # GCNs don't need clamping
            return