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
    dataset, split_idx, evaluator = load_pyg_dataset(args.dataset)
    data = dataset[0]    
    # Attach the node idx to the data
    data['orig_node_idx'] = torch.arange(data.x.shape[0])

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask


    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    dataset_loader = CustomClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=args.data_shuffle, num_workers=args.num_workers,
                           normalize_adj_matrix=args.normalize_adj_matrix)

    # Get model
    log.info('Building model...')
    model = load_full_model(args.load_path, args.gpu_ids)
    model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # Test
    log.info('Testing...')

    # Evaluate, display the stats and save the model
    dev_results = test(model, dataset_loader, device, evaluator, args)

    # Log the metrics
    dev_log_message = ''.join('{} - {}; '.format(k, v) for k, v in dev_results.items())

    log.info(f'Testing - {dev_log_message}')


def test(model, data_loader, device, evaluator, args):

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
            out, _ = model(batch)
            out = out[batch.test_mask]
            labels = batch.y.squeeze(1)[batch.test_mask]

            # Add batch data to the evaluation data
            labels = labels.cpu()
            if len(labels.shape) == 1:
                labels = torch.unsqueeze(labels, -1)
            
            y_true.extend(labels.tolist())
            if args.multi_label_class == True:
                pred = out.cpu().detach().numpy()
                binary_pred = np.zeros(pred.shape).astype('int')
                for i in range(pred.shape[0]):
                    k = labels[i].cpu().detach().numpy().sum().astype('int')
                    topk_idx = pred[i].argsort()[-k:]
                    binary_pred[i][topk_idx] = 1
                y_pred.extend(binary_pred.tolist())
            else:
                y_pred.extend(torch.argmax(out, -1, keepdim=True).cpu().tolist())

    # Evaluate the training results
    results = evaluator.eval({
        'y_true': np.asarray(y_true),
        'y_pred': np.asarray(y_pred)
    })

    return results

if __name__ == '__main__':
    main(get_test_args())
