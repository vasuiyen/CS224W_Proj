# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 05:27:12 2021
"""

import torch 
import copy
import numpy as np

from sklearn.metrics import *

def train(model, dataloaders, optimizer, args):
    """ 
    Train a model with a DeepSnap dataloader 
    
    Note:
    - model.forward() should accept a DeepSnap batch
    - Expects batches to be augmented with attribute 'node_embedding'
    - See DeepSnap.graph.Graph.add_node_attr
    - https://snap.stanford.edu/deepsnap/modules/graph.html
    """
    val_max = 0
    best_model = model

    for epoch in range(1, args["epochs"]+1):
        for i, batch in enumerate(dataloaders['train']):
            
            batch.to(args["device"])
            model.train()
            optimizer.zero_grad()
            label = batch.edge_label.type(torch.FloatTensor).to(args["device"])
            output = model(batch)
            loss = model.loss(output, label)
            loss.backward()
            optimizer.step()

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {}'
            score_train = eval(model, dataloaders['train'], args)
            score_val = eval(model, dataloaders['val'], args)
            score_test = eval(model, dataloaders['test'], args)

            print(log.format(epoch, score_train, score_val, score_test, loss.item()))
            if val_max < score_val:
                val_max = score_val
                best_model = copy.deepcopy(model)
                
    return best_model

def eval(model, dataloader, args):
    """ 
    Evaluate a model with a DeepSnap dataloader 
    Uses accuracy metric. 
    
    TODO: Implement other metrics? 
    
    Note:
    - model.forward() should accept a DeepSnap batch. See net.DeepSnapWrapper
    - Expects batches to be augmented with attribute 'node_embedding'
    - See DeepSnap.graph.Graph.add_node_attr
    - https://snap.stanford.edu/deepsnap/modules/graph.html
    """
    model.eval()
    score = 0

    outputs = []
    labels = []
    for i, batch in enumerate(dataloader['test']):
        labels.append(batch.node_label.detach().cpu().numpy())
        batch.to(args["device"])
        output = torch.sigmoid(model(batch))
        outputs.append(output.detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    preds = np.argmax(outputs, axis=-1)
    score = accuracy_score(labels, preds)
 
    return score