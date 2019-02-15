import newspaper
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    ## build paper and read in articles ##
    five38 = newspaper.build("http://fivethirtyeight.com/", memoize_articles=False)
    article_limit = 100
    five38_articles = five38.articles[0:article_limit]
    raw_text = [0 for art in five38_articles]
    titles = [0 for art in five38_articles]
    urls = [0 for art in five38_articles]

    ## extract raw text ##
    for ind, art in enumerate(five38_articles):
        try:
            art.download()
            art.parse()
        except:
            pass
        else:
            raw_text[ind] = art.text
            titles[ind] = art.title
            urls[ind] = art.url

    while 0 in titles:
        bad_ind = titles.index(0)
        raw_text.pop(bad_ind)
        titles.pop(bad_ind)

    fpath = "/Users/greg/Google Drive/Spring 19/CS6241/fivethirtyeight-topic-modeling/extract-data/"
    fname = "five38_data.csv"
    with open(fpath + fname, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(raw_text)
        writer.writerow(titles)
        writer.writerow(urls)

    ## one stop shop to do tf-idf transform ##
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(raw_text)
    X = X.toarray()
    fname = 'five38_tfidf'
    np.save(file=fpath+fname, arr=X)

    return 1

if __name__ == '__main__':
    main()
