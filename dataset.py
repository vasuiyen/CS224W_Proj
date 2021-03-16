# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 01:11:40 2021

@author: Admin
"""

import torch
import networkx as nx
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_geometric as pyg

from normalization import fetch_normalization
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import convert

class CustomTextDataset(InMemoryDataset):
        
    def __init__(self, root=None, transform=None, pre_transform=None,
                 pre_filter=None, extra_kwargs = {}):
        super(Dataset, self).__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.__indices__ = None
        self.extra_kwargs = extra_kwargs

        if 'download' in self.__class__.__dict__.keys():
            self._download()

        if 'process' in self.__class__.__dict__.keys():
            self._process()
    
    @staticmethod
    def _load_raw_graph(txt_file):
        graph = {}
        with open(txt_file, 'r') as f:
            cur_idx = 0
            for row in f:
                row = row.strip().split()
                adjs = []
                for j in range(1, len(row)):
                    adjs.append(int(row[j]))
                graph[cur_idx] = adjs
                cur_idx += 1
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        normalization="AugNormAdj"
        adj_normalizer = fetch_normalization(normalization)
        adj = adj_normalizer(adj)
        # adj = AmazonAll._sparse_mx_to_torch_sparse_tensor(adj).float()
        return adj
    
    @staticmethod 
    def _sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        tensor = torch.sparse.FloatTensor(indices, values, shape)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
class AmazonAll(CustomTextDataset):
    name: str = "amazon-all"
    dataset_root: str = "dataset/amazon-all"
    
    def __init__(self, portion="0.06", transform=None, pre_transform=None):
        super(AmazonAll, self).__init__(root = self.dataset_root, transform = transform, pre_transform = pre_transform, extra_kwargs= {'portion': portion})
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['adj_list.txt', 'label.txt', 'meta.txt', 'test_idx.txt'] + \
            [f"train_idx-{str(i / 100)}.txt" for i in range(1,10)] + \
            ["train_idx-0.10.txt"]

    @property
    def processed_file_names(self):
        return [f'data-{self.extra_kwargs["portion"]}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        raise NotImplementedError("Download not supported")

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class = self._load_txt_data(self.extra_kwargs['portion'])
        edge_index, edge_attr = convert.from_scipy_sparse_matrix(adj)
        data = Data(
            x = torch.zeros((features.shape[0], 1)),
            # Note: In the original paper, features is a sparse matrix
            # It doesn't work in this framework, so I put a placeholder for now
            edge_index = edge_index, 
            edge_attr = edge_attr,
            y = labels,
            # Note: In this dataset there are 58 independent binary labels
            # our framework does not support that so I chose one of the binary classes 
            # and converted it to 
            idx_train = idx_train, 
            idx_val = idx_val, 
            idx_test = idx_test
        )
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _load_txt_data(self, portion = '0.06'):
        adj = self._load_raw_graph(self.raw_dir + '/adj_list.txt')
        idx_train = list(np.loadtxt(self.raw_dir + '/train_idx-' + str(portion) + '.txt', dtype=int))
        idx_val = list(np.loadtxt(self.raw_dir + '/test_idx.txt', dtype=int))
        idx_test = list(np.loadtxt(self.raw_dir + '/test_idx.txt', dtype=int))
        labels = np.loadtxt(self.raw_dir + '/label.txt')
        with open(self.raw_dir + '/meta.txt', 'r') as f:
            num_nodes, num_class = [int(w) for w in f.readline().strip().split()]
    
        features = sp.identity(num_nodes)
        
        # porting to pytorch
        features = AmazonAll._sparse_mx_to_torch_sparse_tensor(features).float()
        labels = torch.LongTensor(labels)
        #labels = torch.max(labels, dim=1)[1]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
    
        return adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class
        
registry = {
    'amazon-all': AmazonAll
}
    
if __name__ == "__main__":
    dataset = AmazonAll('0.06')
    print(dataset[0].y)
    print(dataset.num_classes)