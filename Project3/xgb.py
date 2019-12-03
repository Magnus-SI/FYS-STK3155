import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class xgbmodelonly:     #may want this for general use, not yet used
    def __init__(self,num_round):
        self.param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic',
                      'nthread': 8, 'eval_metric': 'auc'}
        self.num_round = num_round

    def paramchanger(self, label, value):
        self.param[label] = value

    def fit(self,X,y):
        dfit = xgb.DMatrix(X, label = y)
        self.bst = xgb.train(sepf.param, dfit, self.num_round)

    def predict(X,y):
        dpred = xgb.DMatrix(X, label = y)
        return self.bst.predict(dpred)

class xgbmodel:
    def __init__(self, loader, num_round):
        self.df, self.xlabels, self.ylabels = loader()
        self.param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic',
                      'nthread': 8, 'eval_metric': 'auc'}
        self.num_round = num_round
        self.N = len(self.df)
        self.traintestsimple(train_frac = 0.8)

    def traintestsplit(self, traininds, testinds):
        self.dftrain = self.df.iloc[traininds]
        self.dftest = self.df.iloc[testinds]

    def traintestsimple(self, train_frac):
        inds = self.df.index
        traininds, testinds = np.split(inds, [int(train_frac*self.N)])
        self.traintestsplit(traininds, testinds)

    def traintestpred(self):
        Xtrain = self.dftrain[self.xlabels].values
        ytrain = self.dftrain[self.ylabels].values
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values

        dtrain = xgb.DMatrix(Xtrain, label = ytrain)
        dtest = xgb.DMatrix(Xtest, label = ytest)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        bst = xgb.train(self.param, dtrain, self.num_round, evallist)
        return bst.predict(dtest)

    def roc(self):
        ytest = self.dftest[self.ylabels].values
        ypred = self.traintestpred()
        fpr, tpr, thres = metrics.roc_curve(ytest, ypred)
        plt.plot(fpr, tpr, label = 'XGB')
        plt.legend()
        plt.show()

    def accu(self):
        ytest = self.dftest[self.ylabels].values
        ypred = self.traintestpred()
        accu = metrics.accuracy_score(ytest, np.round(ypred))
        return accu

    def paramchanger(self, label, value):
        self.param[label] = value

if __name__ == "__main__":
    from pulsar import pulsardat
    loader = pulsardat()
    xgbm = xgbmodel(loader, num_round = 10)
    xgbm.roc()
