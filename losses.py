import torch
import torch.nn as nn

def isometricLoss(output, target, type='L1'):
    if type == 'L1':
        loss = nn.L1Loss()
    if type == 'MSE':
        loss = nn.MSELoss()
    if type == 'BCELogit':
        loss = nn.BCEWithLogitsLoss()
    if type == 'BCE':
        loss = nn.BCELoss()
    if type == 'NLL':
        loss = nn.NLLLoss()
    return loss(output, target)
