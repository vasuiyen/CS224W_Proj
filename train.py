# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 05:27:12 2021
Training script

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
import deepsnap

import models
from utils import *
from losses import *
from args import get_train_args


def main(args):

    # Set up logging and devices
    args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
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
    # Download and process data at './dataset/xxx'
    dataset = PygNodePropPredDataset(name = args.dataset, root = 'dataset/')
    labels = dataset[0].y
    split_idx = dataset.get_idx_split() 
    evaluator = Evaluator(name = args.dataset)

    dataset = build_deepsnap_dataset(dataset)
    dataloaders = build_dataloaders(args, dataset, split_idx) 

    # Get model
    log.info('Building model...')
    model = build_model(args, dataset)
    
    model = nn.DataParallel(model)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = load_model(model, args.load_path, args.gpu_ids)
    
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
            train_results = train(model, dataloaders['train'], labels, split_idx['train'], optimizer, device, evaluator, args.loss_type)
            
            # Log the metrics
            train_log_message = ''.join('{} - {}; '.format(k, v) for k, v in train_results.items())
            

            # Visualize in TensorBoard
            for k, v in train_results.items():
                tboard.add_scalar('train/{k}', v, epoch)

            # Evaluate, display the stats and save the model
            dev_results = evaluate(model, dataloaders['valid'], labels, split_idx['valid'], device, evaluator, args.loss_type)

            # Save the model
            saver.save(epoch, model, dev_results[args.metric_name], device)

            # Log the metrics
            dev_log_message = ''.join('{} - {}; '.format(k, v) for k, v in dev_results.items())

            # Visualize in TensorBoard
            for k, v in dev_results.items():
                tboard.add_scalar('eval/{k}', v, epoch)

            log.info(f'Epoch: {epoch} - Training - {train_log_message} - Evaluating - {dev_log_message}')

            progress_bar.update(1)
            progress_bar.set_postfix(eval_loss=dev_results['loss'])


def train(model, data_loader, labels, idx, optimizer, device, evaluator, loss_type):

    model.train()

    with torch.enable_grad():
        for batch in data_loader:
            
            batch = batch.to(device)  
            optimizer.zero_grad()

            labels = labels.to(device)

            # Forward
            out = model(batch)
            loss = isometricLoss(out[idx], torch.squeeze(labels[idx]), loss_type)

            # Backward
            loss.backward()

        optimizer.step()

    results = evaluator.eval({
        'y_true': labels[idx],
        'y_pred': torch.argmax(out[idx], -1, keepdim=True)
    })

    results['loss'] = loss.cpu().item()

    return results


def evaluate(model, data_loader, labels, idx, device, evaluator, loss_type):

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            
            batch = batch.to(device)  

            labels = labels.to(device)

            # Forward
            out = model(batch)
            loss = isometricLoss(out[idx], torch.squeeze(labels[idx]), loss_type)

    results = evaluator.eval({
        'y_true': labels[idx],
        'y_pred': torch.argmax(out[idx], -1, keepdim=True)
    })

    results['loss'] = loss.cpu().item()

    return results

if __name__ == '__main__':
    main(get_train_args())
