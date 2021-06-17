from __future__ import print_function


import numpy
from numpy.core.fromnumeric import argmax
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list, threshold):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :].tolist()
            label = []
            flag = False
            for j in probs_:
                if j >= threshold:
                    flag = True
                    label.append(str(probs_.index(j)))
            if not flag:
                label.append(str(argmax(probs_)))
            all_labels.append(label)
        return all_labels


class Classifier(object):

    def __init__(self, embeddings, ids, clf, threshold):
        self.embeddings = embeddings
        self.ids = ids
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.threshold = threshold

    def train(self, X, Y):#, Y_all):
        self.binarizer.fit(Y)#_all)  #差别貌似不是很大
        X_train = [self.embeddings[self.ids.index(x)] for x in X]
        Y_train = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y_train)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y] 
        Y_ = self.predict(X, top_k_list)
        '''print(type(Y_))
        print(Y_)
        print(Y_[0])
        print(Y_[0][0],type(Y[0][0]))
        print(Y_.shape)
        t = []
        for i in Y_:
            t.append(numpy.argmax(i))
        print(t)
        Y = self.binarizer.transform(Y)
        print(Y)
        print(type(Y))'''
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        print('-------------------')
        #print(1)
        print(results)
        return results
        print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[self.ids.index(x)] for x in X]) #将输入对应为embeddings
        Y = self.clf.predict(X_, top_k_list, self.threshold)
        return Y


    def for_own_data_just_predict(self, X_train, Y_train,X_test):
        state = numpy.random.get_state()
        numpy.random.seed(0)
        '''print(X_train)
        print(Y_train)
        print(type(X_train))
        print(type(Y_train))
        print(len(X_train),len(Y_train))'''
        self.train(X_train, Y_train)
        numpy.random.set_state(state)

        top_1_list = [1 for l in X_test]
        Y_pred = self.predict(X_test, top_1_list)
        return Y_pred
        '''for i in Y_pred:
            print(i)'''


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        y = []
        for i in vec[1:]:
            y.append(int(i))
        y = sorted(y)
        ys = []
        for i in y:
            ys.append(str(i))
        X.append(vec[0])
        Y.append(ys)
    fin.close()
    return X, Y

def read_paper_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1])
    fin.close()
    return X, Y
