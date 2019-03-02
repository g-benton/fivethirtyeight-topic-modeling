import numpy as np
from scipy.sparse import random as sciRand
from scipy.optimize import nnls
from mpl_toolkits.mplot3d import Axes3D
import math
import csv
import sys
sys.path.append("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/")
import matplotlib.pyplot as plt

def update_W(X, W, H, rank=3):
    Wt = np.linalg.lstsq(np.transpose(H), np.transpose(X), rcond=None)
    return W

def update_H(X, W):
    H = np.linalg.lstsq(W, X, rcond=None)[0]
    H[H < 0] = 0
    return H

def ALS(X, rank=3, n_iters=100, seed=66):
    np.random.seed(seed)
    x_norm = np.linalg.norm(X)
    n = X.shape[0]
    m = X.shape[1]
    W = np.random.rand(n, rank)
    H = np.random.rand(rank, m)

    # n_iters = 50
    errors = [None for ii in range(n_iters)]
    for iter in range(n_iters):
        W = update_W(X, W, H, rank)
        H = update_H(X, W)
        errors[iter] = np.divide(np.linalg.norm(X - np.dot(W, H)), x_norm)

    return W, H, errors

if __name__ == '__main__':
    fpath = "/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/"
    fname = "five38_data.csv"
    with open(fpath + fname, newline='') as f:
        reader = csv.reader(f)
        raw_dat = list(reader)

    raw_text = raw_dat[0]
    titles = raw_dat[1]
    classes = raw_dat[2]

    fname = "five38_tfidf.npy"
    X = np.load(fpath + fname)
    X.shape

    W, H, errs = ALS(X)
    plt.plot(errs)
    plt.show()
    plt.plot([math.log(er) for er in errs])
    plt.show()
    W.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=W[:, 0], ys=W[:, 1], zs=W[:,2], c=classes)
    plt.show()
