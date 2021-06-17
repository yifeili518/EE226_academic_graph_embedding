from __future__ import print_function


import numpy
from numpy.core.fromnumeric import argmax
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :].tolist()
            label = []
            #print(probs_)
            flag = False
            for j in probs_:
                if j >=0.1:
                    flag = True
                    label.append(str(probs_.index(j)))
            if not flag:
                label.append(str(argmax(probs_)))
            all_labels.append(label)
        return all_labels


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y):#, Y_all):
        self.binarizer.fit(Y)#_all)  #差别貌似不是很大
        #由于在构图的时候会排除一些孤立点（没有连接边的点），
        #但在training set中会存在一些这种被排除的点 
        #因此我们可以提前去掉training set中的这些点和其标签对
        keys = self.embeddings.keys()
        X_n = []
        Y_n = []
        for x in X:
            if x in keys:
                X_n.append(x)
                Y_n.append(Y[X.index(x)])
        #之后再进行训练
        X_train = [self.embeddings[x] for x in X_n]
        Y_n = self.binarizer.transform(Y_n)
        self.clf.fit(X_train, Y_n)

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
        X_ = numpy.asarray([self.embeddings[x] for x in X]) #将输入对应为embeddings
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        #print(1)
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        '''print(X_train)
        print(Y_train)
        print(X_test)
        print(Y_test)'''
        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    def for_own_data_just_predict(self, X_train, Y_train,X_test_n):
        state = numpy.random.get_state()
        numpy.random.seed(0)
        '''print(X_train)
        print(Y_train)
        print(type(X_train))
        print(type(Y_train))
        print(len(X_train),len(Y_train))'''
        self.train(X_train, Y_train)
        numpy.random.set_state(state)

        top_1_list = [1 for l in X_test_n]
        Y_pred = self.predict(X_test_n, top_1_list)
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
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
