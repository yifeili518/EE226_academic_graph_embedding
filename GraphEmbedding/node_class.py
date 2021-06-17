import csv
import random
from sklearn.linear_model import LogisticRegression

from classification2 import read_node_label, Classifier


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

def store_result(filename, X_train ,Y_train, X_test, Y_pred):
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
            else:
                s = ""
                for k,l in enumerate(Y_pred[X_test.index(x)]):
                    s += l
                    if k != len(Y_pred[X_test.index(x)])-1:
                        s += ' '
                writer.writerow([x,s])
                continue
    #print(tmp)
    #print(count)

        
def node_classification(author_embeddings, author_ids, walk_length,n, vec_size):
    X_train, Y_train = read_node_label('data/author_label.txt')
    X_test = read_node_toTest('data/author_to_pred.txt')
    tr_frac = 0.2
    j = 0.2 #阈值
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=author_embeddings, ids = author_ids, clf=LogisticRegression(solver='lbfgs', max_iter=100000000),threshold = j)
    Y_pred = clf.for_own_data_just_predict(X_train, Y_train, X_test)
    #print(Y_pred)
    store_result('new_result_meta_nc_{}_{}_{}_j={}.csv'.format(walk_length, n, vec_size, j), X_train ,Y_train, X_test, Y_pred)