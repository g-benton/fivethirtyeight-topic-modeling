import numpy as np
from scipy.sparse import random as sciRand
from scipy.optimize import nnls
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import math
import csv
import sys
sys.path.append("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/")
import matplotlib.pyplot as plt

def update_H(X, W, H):
    numer = np.dot(np.transpose(W), X)
    denom = np.dot(np.dot(np.transpose(W), W), H)
    H = np.multiply(H, np.divide(numer, denom))
    # H[H<0] = 0

    return H

def update_W(X, W, H):
    numer = np.dot(X, np.transpose(H))
    denom = np.dot(np.dot(W, H), np.transpose(H))
    W = np.multiply(W, np.divide(numer, denom))
    # W[W<0] = 0

    return W

def mult_update(X, rank=3, n_iters=100, seed=66):
    x_norm = np.linalg.norm(X)
    n = X.shape[0]
    m = X.shape[1]
    np.random.seed(seed)
    W = np.random.rand(n, rank)
    H = np.random.rand(rank, m)

    errors = [None for ii in range(n_iters)]
    for iter in range(n_iters):
        H = update_H(X, W, H)
        W = update_W(X, W, H)
        errors[iter] = np.divide(np.linalg.norm(X - np.dot(W, H)), x_norm)

    return W, H, errors
