# -*- coding: utf-8 -*-
"""

Testing script

"""

import random
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import ClusterData, ClusterLoader

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

import tqdm

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
    evaluator = Evaluator(name = args.dataset)

    split_idx = dataset.get_idx_split() 
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    dataset_loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=args.data_shuffle, num_workers=args.num_workers)

    # Get model
    log.info('Building model...')
    model = load_full_model(args.load_path, args.gpu_ids)
    model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # Test
    log.info('Testing...')

    # Evaluate, display the stats and save the model
    dev_results = test(model, dataset_loader, device, evaluator)

    # Log the metrics
    dev_log_message = ''.join('{} - {}; '.format(k, v) for k, v in dev_results.items())

    log.info(f'Testing - {dev_log_message}')


def test(model, data_loader, device, evaluator):

    model.eval()

    y_true = []
    y_pred = []

    with torch.enable_grad():
        for batch in data_loader:
            
            batch = batch.to(device)
            batch_size = batch.test_mask.sum().item()

            if batch_size == 0:
                continue

            # Forward
            out = model(batch)[batch.test_mask]
            labels = batch.y.squeeze(1)[batch.test_mask]

            # Add batch data to the evaluation data
            y_true.extend(torch.unsqueeze(labels.cpu(), -1).tolist())
            y_pred.extend(torch.argmax(out, -1, keepdim=True).cpu().tolist())

    # Evaluate the training results
    results = evaluator.eval({
        'y_true': np.asarray(y_true),
        'y_pred': np.asarray(y_pred)
    })

    return results

if __name__ == '__main__':
    main(get_test_args())
