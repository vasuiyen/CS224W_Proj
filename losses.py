import torch
import torch.nn as nn

def isometricLoss(output, target, type='L1'):
    if type == 'L1':
        loss = nn.L1Loss()
    if type == 'MSE':
        loss = nn.MSELoss()
    if type == 'BCELogit':
        loss = nn.BCEWithLogitsLoss()
        target = target.to(dtype=torch.float32)
    if type == 'BCE':
        loss = nn.BCELoss()
    if type == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    if type == 'NLL':
        output = torch.log_softmax(output)
        loss = nn.NLLLoss()
    return loss(output, target)

def soft_spectral_loss(A, spectral_radius):
    """ In place of doing a direct maximum, use a smooth approximation to the max """
    assert len(A.shape) == 2
    weights = torch.softmax(A.abs().sum(-1))
    return A * weights

def hard_spectral_loss(A, spectral_radius):
    assert len(A.shape) == 2
    return torch.max(A.abs().sum(-1)) 
   