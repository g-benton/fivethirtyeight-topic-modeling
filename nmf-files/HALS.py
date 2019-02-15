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

def update_W(X, W, H, rank=3):
    ## update column by column ##
    for col in range(rank):

        offset = np.zeros_like(W[:, col])
        for kk in range(rank):
            if kk != col:
                offset += np.dot(W[:, kk], np.dot(H[kk ,:], np.transpose(H[col, :])))


        new_vec = np.divide(np.dot(X, H[col, :]) + offset, np.linalg.norm(H[:, col]))
        new_vec[new_vec < 0] = 0
        W[:, col] = new_vec
        ## NOTE: This method uses previous updates to compute next iterates,
        ## may want to change this if there are convergence issues.
    return W

def update_H(X, W):
    H = np.linalg.lstsq(W, X, rcond=None)[0]
    H[H < 0] = 0
    return H

def HALS(X, rank=3, n_iters=100, seed=66):
    x_norm = np.linalg.norm(X)
    n = X.shape[0]
    m = X.shape[1]
    np.random.seed(seed)
    W = np.random.rand(n, rank)
    H = np.random.rand(rank, m)

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

    fname = "five38_CV.npy"
    X = np.load(fpath + fname)
    X.shape

    W, H, errs = HALS(X, seed=21)
    # plt.plot([math.log(er) for er in errs])
    # plt.show()
    W.shape

    ## set up for plotting stuff ##
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n_cls = len(set(classes))
    clrs = sns.color_palette(n_colors=n_cls)
    c_names = ["Politics", "Sports", "Sci-Tech", "Other"]
    legend_elements = [Line2D([0], [0], marker='o', color=clrs[c], label=c_names[c]) for c in range(n_cls)]
    ax.scatter(xs=W[:, 0], ys=W[:, 1],
        zs=W[:,2], c=classes)


    # cls_plt_inds = [0 for i in range(n_cls)]
    # for c in range(n_cls):
    #     cls_plt_inds[c] = [i for i, cls in enumerate(classes) if cls==c]



    ax.legend()
    plt.show()
