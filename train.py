# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 05:27:12 2021
Training script

"""

import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import ClusterData, ClusterLoader


from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator


import tqdm

import scipy.sparse as sp
from sklearn.metrics import *

from torch.utils.tensorboard import SummaryWriter

from json import dumps
import deepsnap

import models
from utils import *
from losses import *
from args import get_train_args

from models import DataParallelWrapper

def main(args):

    # Set up logging and devices
    args.save_dir = get_save_dir(args.save_dir, args.name + '-' + args.dataset, training=True)
    log = get_logger(args.save_dir, args.name)
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

    num_nodes = data.num_nodes

    # If the node features is a zero tensor with dimension one
    # re-create it here as a (num_nodes, num_nodes) sparse identity matrix
    if data.num_node_features == 1 and torch.equal(data['x'], torch.zeros(data.num_nodes, data.num_node_features)):
        node_features = sp.identity(data.num_nodes/len(dataset_loader))
        node_features = sparse_mx_to_torch_sparse_tensor(node_features).float()
        data.x = node_features 
        
    # Get model
    log.info('Building model...')

    # Create the model, optimizer and checkpoint
    model_class = str_to_attribute(sys.modules['models'], args.name)
    model = model_class(data.x.shape[-1], dataset.num_classes, args, log, orig_num_nodes=num_nodes)
    
    model = DataParallelWrapper(model)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = load_model(model, args.load_path, args.gpu_ids)
    else:
        # Reset parameters only if not loading from checkpoint
        model.reset_parameters()

    model = model.to(device)
    model.train()

    # Get optimizer and scheduler
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, args.learning_rate, 
        weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, args.learning_rate, 
        momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(parameters, args.learning_rate,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(parameters, args.learning_rate,
                                 weight_decay=args.weight_decay)

    # Get saver
    saver = CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Train
    log.info('Training...')
    with tqdm.tqdm(total=args.num_epochs) as progress_bar:
        for epoch in range(args.num_epochs):

            # Train and display the stats
            train_results = train(model, dataset_loader, optimizer, device, evaluator, args)
            
            # Log the metrics
            train_log_message = ''.join('{} - {}; '.format(k, v) for k, v in train_results.items())
            

            # Visualize in TensorBoard
            for k, v in train_results.items():
                tboard.add_scalar(f'train/{k}', v, epoch)

            # Evaluate, display the stats and save the model
            dev_results = evaluate(model, dataset_loader, device, evaluator, args)

            # Save the model
            saver.save(epoch, model, dev_results[args.metric_name], device)

            # Log the metrics
            dev_log_message = ''.join('{} - {}; '.format(k, v) for k, v in dev_results.items())

            # Visualize in TensorBoard
            for k, v in dev_results.items():
                tboard.add_scalar(f'eval/{k}', v, epoch)

            log.info(f'Epoch: {epoch} - Training - {train_log_message} - Evaluating - {dev_log_message}')

            progress_bar.update(1)
            progress_bar.set_postfix(eval_loss=dev_results['loss'])


def train(model, data_loader, optimizer, device, evaluator, args):

    model.train()

    loss_meter = AverageMeter()
    y_true = []
    y_pred = []

    with torch.enable_grad():
        for idx, batch in enumerate(data_loader):

            batch['batch_index'] = idx

            batch = batch.to(device)
            batch_size = batch.train_mask.sum().item()

            if batch_size == 0:
                continue
            
            optimizer.zero_grad()
            
            # Forward
            out, reg = model(batch)
            out = out[batch.train_mask]
            labels = batch.y.squeeze()[batch.train_mask]
            
            # Calculate the loss and do the average
            loss = isometricLoss(out, labels, args.loss_type)
            loss += reg
            loss_meter.update(loss.item(), batch_size)
            
            # Backward
            loss.backward()
            optimizer.step()

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

    results['loss'] = loss_meter.avg

    return results


def evaluate(model, data_loader, device, evaluator, args):

    model.eval()

    loss_meter = AverageMeter()
    y_true = []
    y_pred = []

    with torch.enable_grad():
        for idx, batch in enumerate(data_loader):
            
            batch['batch_index'] = idx

            batch = batch.to(device)
            batch_size = batch.valid_mask.sum().item()

            if batch_size == 0:
                continue

            # Forward
            out, reg = model(batch)
            out = out[batch.valid_mask]
            labels = batch.y.squeeze()[batch.valid_mask]

            # Calculate the loss and do the average
            loss = isometricLoss(out, labels, args.loss_type)
            loss += reg
            loss_meter.update(loss.item(), batch_size)

            # Add batch data to the evaluation data
            labels = labels.cpu()
            if len(labels.shape) == 1:
                labels = torch.unsqueeze(labels, -1)
            
            y_true.extend(labels.tolist())
            if args.multi_label_class == False:
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

    results['loss'] = loss_meter.avg

    return results

if __name__ == '__main__':
    main(get_train_args())
