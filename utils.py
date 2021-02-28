# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 03:55:06 2021
"""

import numpy as np
import scipy.sparse.linalg as linalg

def estimate_spectral_radius(graph):
    """    
    @graph: A networkx.Graph, undirected
    
    @return: An upper bound on the spectral radius
    
    Notes:
    - spectral radius is another name for "Perron-Frobenius eigenvalue"
    (i.e. largest eigenvalue of a real, square matrix).
    - The estimation relies on the Gershgorin Circle Theorem. 
    https://en.wikipedia.org/wiki/Gershgorin_circle_theorem    
    - For graph adjacency matrices, the theorem says:
    spectral radius <= the maximum node degree
    """
    # TODO
    return max(graph.degree)    

def compute_spectral_radius(A, l = None):
    """
    @param A: 
        Numpy matrix. Must be real-valued and square
    
    @param l: 
        initial estimate of the spectral radius of A.
        Recommended to use estimate_spectral_radius for this purpose. 
    
    Based on: 
    https://scicomp.stackexchange.com/questions/21727/numerical-computation-of-perron-frobenius-eigenvector
    """
    v = np.ones(A.shape[0])
    eigenvalues, eigenvectors = linalg.eigs(A, k=2, sigma=l, which='LM', v0=v)