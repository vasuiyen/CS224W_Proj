import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Function

class ImplicitFunction(Function):
    # Do this the forward iterative iteration as a nn.Function oposed to a nn.Module to improve GPU memory efficiency by 10x for 300 iterations
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, aggr, attn_layer, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, aggr, attn_layer, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):

        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        grad_z, err, status, _ = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, aggr='None', attn_layer='None', mitr=bw_mitr, trasposed_A=True)
        #grad_z.clamp_(-1,1)

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, aggr, attn_layer, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        #X = B if X is None else X

        err = 0
        status = 'max itrs reached'	
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                if aggr == 'sum':
                   dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]			   
                elif aggr == 'mean':
                   dphi = torch.autograd.grad(torch.mean(X_new), Z, only_inputs=True)[0]
                elif aggr == 'attn':
                   dphi = torch.autograd.grad(ImplicitFunction.attn_aggr(attn_layer, X_new), Z, only_inputs=True)[0]
        return X_new, err, status, dphi

    @staticmethod
    def attn_aggr(attn_layer, X_new):
        X_out = attn_layer(X_new)
        out = torch.softmax(X_out, dim=0)			
        return torch.sum(out)
