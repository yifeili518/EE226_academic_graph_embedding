
import numpy as np
import random


from classification1 import read_node_label, Classifier
from ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import csv

from link_pred1 import link_prediction


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
        count = 0
        tmp = []
        for x in X_test:
            if x in X_train:
                count += 1
                s = ""
                for k, l in enumerate(Y_train[X_train.index(x)]):
                    s += l
                    if k != len(Y_train[X_train.index(x)])-1:
                        s += ' '
                writer.writerow([x, s])
                continue
            if x in X_test_n:
                count += 1
                s = ""
                for k,l in enumerate(Y_pred[X_test_n.index(x)]):
                    s += l
                    if k != len(Y_pred[X_test_n.index(x)])-1:
                        s += ' '
                writer.writerow([x,s])
                continue
            if x not in X_train and x not in X_test_n:
                writer.writerow([x, '{}'.format(random.randint(0,9))])
                tmp.append(x)

        
def predict(embeddings,walk_length,num_walks,p,q):
    X_train, Y_train = read_node_label('data/author_label.txt')
    X_test = read_node_toTest('data/author_to_pred.txt')
    tr_frac = 0.2
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='lbfgs', max_iter=100000000))
    X_test_n = []
    for x in X_test:
        if x in embeddings.keys():
            X_test_n.append(x)
    Y_pred = clf.for_own_data_just_predict(X_train, Y_train, X_test_n)
    store_result('result_node2vec_{}_{}_{}_{}_nc.csv'.format(walk_length,num_walks,p,q), X_train ,Y_train, X_test, X_test_n, Y_pred)




if __name__ == "__main__":
    G=nx.read_edgelist('data/co_citation.txt',
                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])
    walk_length = 10
    num_walks = 80
    p = 0.25
    q = 4

    model = Node2Vec(G, walk_length=walk_length, num_walks=num_walks,
                     p=p, q=q, workers=1, use_rejection_sampling=0) #原来为10 80 0.25 4
    model.train(window_size = 5, iter = 3)
    embeddings=model.get_embeddings()

    print("walk_length:{},num_walks:{},p={},q={}".format(walk_length,num_walks,p,q))
    predict(embeddings,walk_length,num_walks,p,q)
    link_prediction(embeddings, walk_length, num_walks, p, q)
