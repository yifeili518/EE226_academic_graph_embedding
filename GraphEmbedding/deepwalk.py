import csv
import numpy as np

from classification1 import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


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
def store_result(filename, X_train ,Y_train, X_test, X_test_n, Y_pred):
    with open(filename,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['author_id','labels'])

        for x in X_test:
            if x in X_train:
                s = ""
                for k, l in enumerate(Y_train[X_train.index(x)]):
                    s += l
                    if k != len(Y_train[X_train.index(x)])-1:
                        s += ' '
                writer.writerow([x, s])
                continue
            if x in X_test_n:
                s = ""
                for k,l in enumerate(Y_pred[X_test_n.index(x)]):
                    s += l
                    if k != len(Y_pred[X_test_n.index(x)])-1:
                        s += ' '
                writer.writerow([x, s])
                continue
            if x not in X_train and x not in X_test_n:
                writer.writerow([x, '2 5'])

        
def predict(embeddings):
    X_train, Y_train = read_node_label('data_preprocessing/author_label.txt')
    X_test = read_node_toTest('data_preprocessing/author_to_pred.txt')
    tr_frac = 0.2
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    X_test_n = []
    for x in X_test:
        if x in embeddings.keys():
            X_test_n.append(x)
    Y_pred = clf.for_own_data_just_predict(X_train, Y_train, X_test_n)
    #print(Y_pred)
    store_result('result.csv', X_train ,Y_train, X_test, X_test_n, Y_pred)




if __name__ == "__main__":
    G = nx.read_edgelist('data_preprocessing/co_citation.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    predict(embeddings)
