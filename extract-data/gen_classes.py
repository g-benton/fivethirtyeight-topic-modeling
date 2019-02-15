## just manually labeling some entries to see what's happening ##
import csv

fpath = "/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/"
fname = "five38_data.csv"
with open(fpath + fname, newline='') as f:
    reader = csv.reader(f)
    raw_dat = list(reader)

raw_text = raw_dat[0]
titles = raw_dat[1]
classes = [0 for tt in titles]
titles[0:10]
classes[0:10] = [0,0,0,1,1,3,0,3,0,0]
titles[10:20]
classes[10:20] = [1,0,1,0,3,1,1,1,0]

titles[20:30]

classes[20:30] = [0,0,0,0,0,0,0,0,0,0]
titles[30:40]

classes[30:40] = [1,3,0,0,0,0,0,0,2,0]
titles[40:50]
classes[40:50] = [1,1,2,0,0,0,1,0,0,1]
titles[50:60]
classes[50:60] = [1,1,0,3,0,0,3,0,0,0]
titles[60:70]
classes[60:70] = [0,0,2,2,1,2,0,0,0,0]
titles[70:80]
classes[70:80] = [0,0,0,0,0,2,2,2,2,2]
titles[80:90]
classes[80:90] = [2,2,0,2,2,2,1,2,2,2]
titles[90:100]
classes[90:100] = [1,1,0,1,2,0,2,2,0,1]


fpath = "/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/"
fname = "five38_data.csv"
with open(fpath + fname, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(raw_text)
    writer.writerow(titles)
    writer.writerow(classes)
