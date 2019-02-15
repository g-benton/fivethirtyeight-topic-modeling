import csv

with open("/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/cnn_data.csv", newline='') as f:
    reader = csv.reader(f)
    raw_dat = list(reader)

raw_text = raw_dat[0]
titles = raw_dat[1]
