import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers.core import Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import numpy as np
import csv
from numpy.core.fromnumeric import argmax
import os
import tensorflow as tf
from tensorflow import keras

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


def read_node_toTest(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        head = next(reader)
        data = list(reader)
        X_test = []
        for row in data:
            X_test.append(row[0])
        return X_test


def deep_model(feature_dim,label_dim):
    from keras.models import Sequential
    from keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.losses
    from tensorflow.keras.applications.resnet import ResNet50
    from keras import backend as K
    model = Sequential()
    #model = ResNet50(weights=None, classes=label_dim)
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dropout(0.1))
    model.add(Dense(200, input_dim=feature_dim,activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    adam = Adam(learning_rate=1e-3)

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

walk_length = 150
num_walk = 20
vec_size = 64

model_file = 'models/model_{}_{}_{}.h5'.format(walk_length, num_walk, vec_size)
def train_and_predict(X_train, y_train, X_test):
    if not os.path.exists(model_file):
        feature_dim = X_train.shape[1]
        label_dim = y_train.shape[1]
        model = deep_model(feature_dim,label_dim)
        #model.summary()
        import keras.backend as K     
        from keras.callbacks import LearningRateScheduler

        def scheduler(epoch):
            # 每隔100个epoch，学习率减小为原来的1/5
            if epoch % 100 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.2)
                print("lr changed to {}".format(lr * 0.2))
            return K.get_value(model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)
        model.fit(X_train, y_train, epochs=1000, callbacks=[reduce_lr])#,validation_data=(X_test,y_test))
        model.save(model_file)
    else:
        model = keras.models.load_model(model_file)

    X_test_embeddings = []
    for x in X_test:
        X_test_embeddings.append(author_embeddings[author_ids.index(x)])
    X_test_embeddings = np.array(X_test_embeddings)   
    Y_predict = model.predict(X_test_embeddings)
    print(type(Y_predict))
    print(Y_predict.shape)
    print(Y_predict[0])

    return Y_predict

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
    

author_embeddings = np.load("embeddings_array/author_embeddings_{}_{}_{}.npy".format(walk_length, num_walk, vec_size)).tolist()
author_ids = np.load("embeddings_array/author_ids_{}_{}_{}.npy".format(walk_length, num_walk, vec_size)).tolist()
X_train, Y_train = read_node_label('data/author_label.txt')

X_train_embeddings = []
for x in X_train:
    X_train_embeddings.append(author_embeddings[author_ids.index(x)])
X_train_embeddings = np.array(X_train_embeddings)
Y_train_bi = []
for y in Y_train:
    bi = np.zeros(10)
    for label in y:
        bi[int(label)] = 1
    Y_train_bi.append(bi)
Y_train_bi = np.array(Y_train_bi)


threshold = 0.08

X_test = read_node_toTest('data/authors_to_pred.csv')
Y_predict = train_and_predict(X_train_embeddings,Y_train_bi, X_test).tolist()
Y_pred = []
for y in Y_predict:
    multilabels = []
    flag = False
    for prob in y:
        if prob >= threshold:
            flag  = True
            multilabels.append(str(y.index(prob)))
    if not flag:
        multilabels.append(str(argmax(y)))

    Y_pred.append(multilabels)


store_result('1_result_meta_nc_{}_{}_{}_j={}_newclassi.csv'.format(walk_length, num_walk, vec_size, threshold), X_train ,Y_train, X_test, Y_pred)
