
from networkx.algorithms import wiener
import numpy as np
import csv

from classification1 import read_node_label, Classifier
from ge import SDNE
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
                writer.writerow([x, '5'])

        
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



def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1],
                    label=c)  # c=node_colors)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X_test = read_node_toTest('data_preprocessing/author_to_pred.txt')

    G1 = nx.read_edgelist('data_preprocessing/author_citation.txt', #../data/wiki/Wiki_edgelist.txt
                         create_using=nx.DiGraph(), nodetype=None)#, data=[('weight', int)]
    print(G1.number_of_nodes())
    print(G1.number_of_edges())

    G2 = nx.read_edgelist('data_preprocessing/coauthor.txt',create_using=nx.DiGraph(), nodetype=None)
    print(G2.number_of_nodes())
    print(G2.number_of_edges())

    G3 = nx.read_edgelist('data_preprocessing/co_citation.txt',create_using=nx.DiGraph(), nodetype=None)
    print(G3.number_of_nodes())
    print(G3.number_of_edges())
    model = SDNE(G3, hidden_size=[256, 128],)
    model.train(batch_size=3000, epochs=15, verbose=2)#初始值3000 40 2
    embeddings = model.get_embeddings()
    print(embeddings.keys())
    print(len(embeddings.keys()))


    predict(embeddings)

