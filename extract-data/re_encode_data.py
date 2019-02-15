import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys

sys.path.append("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/")

def main():
    ## read in extracted papers ##
    fpath = "/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/"
    fname = "five38_data.csv"

    with open(fpath + fname, newline='') as f:
            reader = csv.reader(f)
            raw_dat = list(reader)

    raw_text = raw_dat[0]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(raw_text)
    X = X.toarray()
    fname = 'five38_CV'
    np.save(file=fpath+fname, arr=X)


    return 1


if __name__ == '__main__':
    main()
