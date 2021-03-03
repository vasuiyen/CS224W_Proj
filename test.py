# -*- coding: utf-8 -*-
"""

Testing script

"""

import sys
import copy
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch_geometric.data import DataLoader

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator


import tqdm

from collections import OrderedDict
from sklearn.metrics import *

from torch.utils.tensorboard import SummaryWriter

from json import dumps

from utils import *
from losses import *
from args import get_test_args


def main(args):

    # Set up logging and devices
    args.save_dir = get_save_dir(args.save_dir, 'test', training=True)
    log = get_logger(args.save_dir, 'test')
    tboard = SummaryWriter(args.save_dir)
    device, args.gpu_ids = get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get data loader
    log.info('Building dataset...')
    # Download and process data at './dataset/xxx'
    dataset = PygNodePropPredDataset(name = args.dataset, root = 'dataset/')
    labels = dataset[0].y
    split_idx = dataset.get_idx_split() 
    evaluator = Evaluator(name = args.dataset)

    dataset = build_deepsnap_dataset(dataset)
    dataloaders = build_dataloaders(args, dataset, split_idx) 

    # Get model
    log.info('Building model...')
    model = load_full_model(args.load_path, args.gpu_ids)
    model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # Test
    log.info('Testing...')

    # Evaluate, display the stats and save the model
    dev_results = test(model, dataloaders['test'], labels, split_idx['test'], device, evaluator)

    # Log the metrics
    dev_log_message = ''.join('{} - {}; '.format(k, v) for k, v in dev_results.items())

    log.info(f'Testing - {dev_log_message}')


def test(model, data_loader, labels, idx, device, evaluator):

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            
            batch = batch.to(device)  

            labels = labels.to(device)

            # Forward
            out = model(batch)

    results = evaluator.eval({
        'y_true': labels[idx],
        'y_pred': torch.argmax(out[idx], -1, keepdim=True)
    })

    return results

if __name__ == '__main__':
    main(get_test_args())
