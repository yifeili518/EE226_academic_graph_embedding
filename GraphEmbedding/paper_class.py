import csv

import random
from pandas.io.parsers import read_csv
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

from classification2 import read_node_label, Classifier, read_paper_label


def read_paper_toTest(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        X = []
        for i in data:
            X.append(i[0])
    return X

def read_node_toTest(filename, skip_head=True):
    fin = open(filename, 'r')
    X_test = []
    if skip_head:
        l = fin.readline()
    lines = fin.readlines()
    for i in lines:
        i = i.strip('\n')
        X_test.append(i)
    fin.close()
    return X_test


def store_result(filename, author_X_train ,author_Y_train, author_X_test, X_train, Y_train, Y_pred):
    Y_train = Y_train.tolist()
    with open(filename,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['author_id','labels'])

        authorship = [[] for i in range(42614)]
        with open('data/authorship.csv') as rf:
            reader = csv.reader(rf)
            data = list(reader)
            for row in data:
                author = row[0]
                paper = row[1]
                authorship[int(author)].append(paper)

        for x in author_X_test:
            if x in author_X_train:
                s = ""
                for k, l in enumerate(author_Y_train[author_X_train.index(x)]):
                    s += l
                    if k != len(author_Y_train[author_X_train.index(x)])-1:
                        s += ' '
                writer.writerow([x, s])
                continue
            else:
                papers = authorship[int(x)] # author写的papers
                labels = [] 
                for paper in papers:
                    if paper in X_train:
                        label = Y_train[X_train.index(paper)]
                    else:
                        label = Y_pred[int(paper)]
                    if label not in labels:
                        labels.append(label) #根据paperid得到paper的label 
                s = ""
                for k,l in enumerate(labels):
                    s += l
                    if k != len(labels)-1:
                        s += ' '
                writer.writerow([x,s])
                continue
    #print(tmp)
    #print(count)

        
def paper_classification(paper_embeddings, paper_ids, walk_length,n):
    X_train, Y_train = read_paper_label('data/paper_label.txt')
    X_train_embeddings = []
    for x in X_train:
        X_train_embeddings.append(paper_embeddings[paper_ids.index(x)])
    X_train_embeddings = np.array(X_train_embeddings)
    Y_train = np.array(Y_train)
    print(X_train_embeddings.shape)
    print(Y_train.shape)
    clf = LogisticRegressionCV(
        Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1000
    )
    clf.fit(X_train_embeddings, Y_train)
    print(accuracy_score(Y_train, clf.predict(X_train_embeddings)))
    X_test = read_paper_toTest('data/papers.csv')
    #print(X_test)
    print(len(X_test))
    X_test_embeddings = []
    for x in X_test:
        X_test_embeddings.append(paper_embeddings[paper_ids.index(x)])
    X_test_embeddings = np.array(X_test_embeddings)
    Y_pred = clf.predict(X_test_embeddings)
    print(Y_pred.shape)
    print(Y_pred.tolist())
    author_X_train, author_Y_train = read_node_label('data/author_label.txt')
    author_X_test = read_node_toTest('data/author_to_pred.txt')
    store_result('result_meta_nc_{}_{}_paper_class_to_author.csv'.format(walk_length, n), author_X_train ,author_Y_train, author_X_test, X_train, Y_train, Y_pred)