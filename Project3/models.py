import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf

class XGBoost:     #may want this for general use, not yet used
    def __init__(self,num_round):
        self.param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic',
                      'nthread': 8, 'eval_metric': 'auc'}
        self.num_round = num_round

    def paramchanger(self, label, value):
        self.param[label] = value

    def fit(self,X,y):
        dfit = xgb.DMatrix(X, label = y)
        self.bst = xgb.train(self.param, dfit, self.num_round)

    def predict(self, X):
        dpred = xgb.DMatrix(X)
        return self.bst.predict(dpred)

class NNmodel:
    def __init__(self):
        self.param = {
                      'layers': [50,2],
                      'activations': ['relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': ['AUC'],
                      'epochs': 10,
                      'batch_size': 32
                     }
        self.initmodel()

    def initmodel(self):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(int(i), activation = j)\
                for i,j in zip(self.param['layers'], self.param['activations'])
            ])

        model.compile(
            optimizer = self.param['optimizer'],
            loss = self.param['loss'],
            metrics = self.param['metrics']
        )
        self.model = model

    def paramchanger(self, label, value):
        self.param[label] = value
        if label not in ['epochs', batch_size]:
            self.initmodel()

    def expanddim(self,y):
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fit(self,X,y):
        if len(y.shape)==1:
            y = self.expanddim(y)
        self.model.fit(X, y,
                      epochs = self.param['epochs'],
                      batch_size = self.param['batch_size']
                      )

    def predict(self, X):
        return self.model.predict(X)[:,0]

class analyze:
    def __init__(self, models, loader):
        """
        models: list of models, e.g. [NN,xgboost]
        """
        self.models = models
        self.df, self.xlabels, self.ylabels = loader()
        self.N = len(self.df)
        self.traintestsimple(train_frac = 0.8)

    def traintestsplit(self, traininds, testinds):
        self.dftrain = self.df.iloc[traininds]
        self.dftest = self.df.iloc[testinds]

    def traintestsimple(self, train_frac):
        inds = self.df.index
        traininds, testinds = np.split(inds, [int(train_frac*self.N)])
        self.traintestsplit(traininds, testinds)

    def traintestpred(self, cost):
        Xtrain = self.dftrain[self.xlabels].values
        ytrain = self.dftrain[self.ylabels].values
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values
        scores = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            model.fit(Xtrain, ytrain)
            ypred = model.predict(Xtest)
            print(ytest.shape, ypred.shape)
            scores[i] = metrics.accuracy_score(ytest, np.round(ypred))
        print(scores)

if __name__ == "__main__":
    from pulsar import pulsardat
    loader = pulsardat()
    model1 = xgbmodelonly(num_round = 100)
    model2 = NNmodel()
    models = [model1, model2]
    A = analyze(models, loader)
    A.traintestpred("ok")
