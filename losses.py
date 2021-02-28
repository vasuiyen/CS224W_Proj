import torch
import torch.nn as nn

def isometricLoss(output, target, type='L1'):
    if type == 'L1':
        loss = nn.L1Loss()
    if type == 'MSE':
        loss = nn.MSELoss()
    return loss(output, target)
