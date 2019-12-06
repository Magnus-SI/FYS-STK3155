import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

class ay:
    def __init__(self, model, loader):
        """
        models: list of models, e.g. [NN,xgboost]
        """
        self.dftrain, self.dftest, self.xlabels, self.ylabels = loader()
        self.model = model

    def traintestpred(self, cost):
        Xtrain = self.dftrain[self.xlabels].values
        ytrain = self.dftrain[self.ylabels].values
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values
        self.model.fitCNN(Xtrain, ytrain)
        #return self.model.CNNoutlayer1(Xtest)
        self.model.fitxgb(Xtrain, ytrain)

class CNNxgb:
    def __init__(self, num_round_xgb):
        self.paramxgb = {'max_depth': 3,
                      'eta': 1,
                      'objective': 'binary:logistic',
                      'nthread': 8,
                      'eval_metric': 'auc'}

        self.paramCNN = {'CNNfilters': [64],
                      'DenseLayers' : [512,2],
                      'kernel_size': 5,
                      'activations': ['relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': [tf.keras.metrics.AUC(curve='PR'), 'FalseNegatives'],
                      'epochs': 1,
                      'batch_size': 32,
                      'input_len': 3197,
                      'pool_size': 5,
                      'strides': 2
                      }
        self.num_round_xgb = num_round_xgb
        self.initCNN()

    def initCNN(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(filters=self.paramCNN['CNNfilters'][0],\
                  kernel_size= self.paramCNN['kernel_size'],\
                  activation = self.paramCNN['activations'][0],\
                  input_shape=(self.paramCNN['input_len'],1),
                  name = 'CNN1'))
        model.add(tf.keras.layers.MaxPool1D())
        for fnum in self.paramCNN['CNNfilters'][1:]:
            model.add(tf.keras.layers.Conv1D(filters=fnum,\
                      kernel_size= self.paramCNN['kernel_size'],
                      activation = self.paramCNN['activations'][0]))
            #model.add(tf.keras.layers.Conv1D(filters=fnum,\
            #          kernel_size= self.paramCNN['kernel_size'],\
            #          activation = self.paramCNN['activations'][0]))
            model.add(tf.keras.layers.MaxPool1D(pool_size = self.paramCNN['pool_size'],
                                                strides = self.paramCNN['strides']))

        model.add(tf.keras.layers.Flatten())

        for N in self.paramCNN['DenseLayers'][:-1]:
            model.add(tf.keras.layers.Dense(N,self.paramCNN['activations'][0]))
        model.add(tf.keras.layers.Dense(self.paramCNN['DenseLayers'][-1],self.paramCNN['activations'][-1]))

        model.compile(
            optimizer = self.paramCNN['optimizer'],
            loss = self.paramCNN['loss'],
            metrics = self.paramCNN['metrics']
        )
        self.CNN = model
        self.interm = tf.keras.models.Model(inputs = model.input,
                                            outputs = model.get_layer('CNN1').output)


    def paramchanger(self, label, value, paramtype = 'xgb'):
        if paramtype == 'xgb':
            self.paramxgb[label] = value
        else:
            self.paramCNN[label] = value
            if label not in ['epochs', 'batch_size']:
                self.initmodel()

    def expanddim_X(self,X):
        xnew = np.zeros(list(X.shape)+ [1])
        xnew[:,:,0] = X
        return xnew

    def expanddim_y(self,y):
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fitCNN(self,X,y):
        """
        fit CNN
        """
        if len(y.shape)==1:
            y = self.expanddim_y(y)
        if len(X.shape)==2:
            X = self.expanddim_X(X)

        self.CNN.fit(X, y,
                      epochs = self.paramCNN['epochs'],
                      batch_size = self.paramCNN['batch_size'],
                      verbose = True
                      )

    def CNNoutlayer1(self, X):
        """
        return intermediate output from CNN
        """
        if len(X.shape)==2:
            X = self.expanddim_X(X)
        ok = self.CNN.predict(X)
        return self.CNN.layers[0].output
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name)
        # with tf.Session() as sess:
        #     FC1 = tf.get_default_graph().get_tensor_by_name('CNN1:0')
        #     FC1_values = sess.run(FC1, feed_dict = {x: X})
        # return FC1_values

    def fitxgb(self, X, y):
        """
        fit xgb to intermediate output from CNN
        """
        if len(y.shape)==1:
            y = self.expanddim_y(y)
        if len(X.shape)==2:
            X = self.expanddim_X(X)
        # Xlayer = self.CNNoutlayer1(X)
        # with tf.Session() as sess:
        #     X = Xlayer.eval(feed_dict = {Xlayer:X})
        X = self.interm.predict(X)
        print(X.shape)
        dfit = xgb.DMatrix(X, label = y)
        self.bst = xgb.train(self.param, dfit, self.num_round)

    def predict(self, X):
        if len(y.shape)==1:
            y = self.expanddim_y(y)
        if len(X.shape)==2:
            X = self.expanddim_X(X)
        #X = self.CNNoutlayer1(X)
        dpred = xgb.DMatrix(X)
        return self.bst.predict(dpred)

if __name__ == "__main__":
    from CNN import exodat
    loader = exodat()
    model = CNNxgb(10)
    A = ay(model, loader)
    ok = A.traintestpred('ok')  #still doesn't work but hopefully it will soon.
