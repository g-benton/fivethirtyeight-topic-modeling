import sys
import csv
import numpy as np

sys.path.append("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/nmf-files/")
from ALS import *
sys.path.append("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/main-files/")
# sys.path.append("./nmf-files/")
# sys.path.append("./main-files/")
# sys.path.append("./extract-data/")
from plotting import plotting

def main():
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

    W, H, errs = ALS(X, n_iters=100, rank=3,seed=21)
    plotting(W, classes=classes)

if __name__ == '__main__':
    main()
