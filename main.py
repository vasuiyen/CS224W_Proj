# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 06:05:41 2021
"""


import argparse
import torch_geometric
import deepsnap

from train import train, eval
from net import *

def parse_args():
    parser = argparse.ArgumentParser('Script to train IGNN on torch_geometric dataset')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="karate")
    parser.add_argument("--task", type=str, default="node")
    
def load_pyg_dataset(dataset_name):
    """ Load a torch_geometric dataset by name """
    pyg_dataset = None
    if dataset_name == "karate":
        pyg_dataset = torch_geometric.datasets.KarateClub()
    raise Exception("Dataset name not recognized")
    
def build_deepsnap_dataset(args):
    """ Convert a torch geometric dataset to a DeepSnap dataset"""
    pyg_dataset = load_pyg_dataset(args.dataset)
    graphs = deepsnap.dataset.GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task=args.task)

def build_dataloaders(dataset):
    datasets = {}
    dataset["train"], datasets["val"], datasets["test"] = dataset.split(
        transductive=args.transductive, 
        split_ratio=[args.train_fraction, args.val_fraction, args.test_fraction])
    dataloaders = {split: torch.utils.data.DataLoader(
                ds, collate_fn=deepsnap.batch.Batch.collate([]),
                batch_size=args.batch_size, shuffle=(split=='train'))
                for split, ds in datasets.items()}
    return dataloaders
    
def build_model(args, dataset):
    """
    
    Note: Hardcoded to use node features for now
    """
    node_channels = dataset.num_node_features
    hidden_channels = args.hidden_dim
    prediction_channels = dataset.num_node_labels
    model = RecurrentGraphNeuralNet(node_channels, hidden_channels, prediction_channels)

def main():
    args = parse_args()
    dataset = build_deepsnap_dataset(args)
    dataloaders = build_dataloaders(dataset)
    model = build_model(args, dataset)    
    
if __name__ == "__main__":
    main()

    