import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from pulsar import pulsardat
from time import time

class LinSVC:
    def __init__(self):
        self.clf = LinearSVC()

    def fit(self,X,y):
        self.clf.fit(X, y)

    def predict(self,X):
        return self.clf.predict(X)

class LogReg:
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self,X,y):
        self.clf.fit(X, y)

    def predict(self,X):
        return self.clf.predict_proba(X)[:,1]

class NNxgb:
    def __init__(self, num_round_xgb):
        self.paramxgb = {'max_depth': 3,
                      'eta': 1,
                      'objective': 'binary:logistic',
                      'nthread': 4,
                      'eval_metric': 'auc',
                      'booster': 'dart'}

        self.paramNN = {
                      'layers': [128,2],
                      'activations': ['relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': ['AUC'],
                      'epochs': 20,
                      'batch_size': 32
                     }
        self.num_round_xgb = num_round_xgb
        self.initNN()

    def initNN(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation = 'relu',
                                        input_shape = (8,), name = 'layer1'))
        model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
        # model = tf.keras.models.Sequential([
        #         tf.keras.layers.Dense(int(i), activation = j, name = "%ilayer"%i)\
        #         for i,j in zip(self.paramNN['layers'], self.paramNN['activations'])
        #     ])

        model.compile(
            optimizer = self.paramNN['optimizer'],
            loss = self.paramNN['loss'],
            metrics = self.paramNN['metrics']
        )
        self.NN = model
        self.interm = tf.keras.models.Model(inputs = model.input,
                                            outputs = model.get_layer('layer1').output)

    def paramchanger(self, label, value, paramtype = 'xgb'):
        if paramtype == 'xgb':
            self.paramxgb[label] = value
        else:
            self.paramCNN[label] = value
            if label not in ['epochs', 'batch_size']:
                self.initmodel()

    def expanddim(self,y):
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fitNN(self, X, y):
        if len(y.shape)==1:
            y = self.expanddim(y)
        self.NN.fit(X, y,
                      epochs = self.paramNN['epochs'],
                      batch_size = self.paramNN['batch_size'],
                      verbose = True
                      )

    def fitxgb(self, X, y):
        X = self.interm.predict(X)
        dfit = xgb.DMatrix(X, label = y)
        self.bst = xgb.train(self.paramxgb, dfit, self.num_round_xgb)

    def fit(self, X,y):
        self.fitNN(X,y)
        self.fitxgb(X,y)

    def predict(self, X):
        X = self.interm.predict(X)
        dpred = xgb.DMatrix(X)
        return self.bst.predict(dpred)

class XGBoost:     #may want this for general use, not yet used
    def __init__(self,num_round):
        self.param = {'max_depth': 3,
                      'eta': 1,
                      'objective': 'binary:logistic',
                      'nthread': 8,
                      'eval_metric': 'auc',
                      'booster': 'dart',
                      'verbosity': 1
                     }
        self.num_round = num_round

    def paramchanger(self, label, value):
        if label == 'num_round':
            self.num_round = value
        else:
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
                      'layers': [128,2],
                      'activations': ['relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': ['AUC'],
                      'epochs': 20,
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
        if label not in ['epochs', 'batch_size']:
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
        inds = self.df.index.values
        np.random.shuffle(inds)
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
            precision, recall, thresholds = metrics.precision_recall_curve(ytest, ypred)
            scores[i] = metrics.auc(recall, precision)
            #scores[i] = metrics.accuracy_score(ytest, np.round(ypred))
        print(scores)
        return scores[0]

    def plot_PR(self):
        """
        Plots the PR (precision-recall) curve for the model using the train data
        """
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values

        base_val = np.count_nonzero(ytest)/len(ytest)
        plt.plot([0,1],[base_val,base_val], 'r--', label = "Random guessing")

        for model in self.models:
            ypred = model.predict(Xtest)
            name = type(model).__name__
            if name == "XGBoost":
                name = model.param['booster']
            if len(ypred.shape)==2:
                ypred=ypred[:,0]
            precision, recall, thresholds = metrics.precision_recall_curve(ytest, ypred)
            plt.plot(precision,recall,label=name)
        plt.legend()
        plt.show()

    def optparamfinder(self, labels, values, Nloops):
        """
        labels: list of labels
        values: list of arrays containing corresponding values
        """

        if len(self.models)!=1:
            print("only works with one model at a time")
            return
        model = self.models[0]
        optinds = np.zeros((Nloops, len(labels))).astype(int)
        opterrs = np.zeros(Nloops)
        for i in range(Nloops):                     #goes through Nloops loops
            for j,arr in enumerate(values):           #goes through the different parameters
                err_arr = np.zeros(len(arr))
                for k,val in enumerate(arr):        #goes through the values in a parameter
                    print(i,j,k)
                    model.paramchanger(labels[j], val)                #update parameter
                    #err_arr[k] = self.traintesterr(testerr = True)
                    #evaluate kfold error with k=2,3,4,5with updated parameter:
                    err_arr[k] = self.traintestpred("ok")

                optind = np.argmax(err_arr)     #index of optimal value of the parameter
                optinds[i,j] = optind           #store index
                model.paramchanger(labels[j], arr[optind])    #change to optimal value
            opterrs[i] = np.max(err_arr)        #error for the foudn optimal values this loop
        return optinds, opterrs

def xgbtreeopter(ttype = 'dart'):
    """
    ttype: dart or gbtree
    """
    model = XGBoost(num_round = 10)
    model.paramchanger('booster', ttype)
    models = [model]
    loader = pulsardat()
    A = analyze(models, loader)

    labels = ['max_depth',
              'eta',
              'num_round',
              'subsample',
              'gamma',
              'lambda',
              'alpha',
              'scale_pos_weight'
             ]

    values = [np.arange(1, 5),
              np.array([0.25, 0.5, 0.75]),
              np.array([3, 5, 10, 15, 30]),
              np.array([0.5, 0.75, 1]),
              np.array([0, 0.5, 1]),
              np.array([1, 1.5]),
              np.array([0, 0.5]),
              np.array([0.1, 1, 5, 10]),
              ]

    if ttype == 'dart':
        labels.append('rate_drop')
        values.append(np.array([0.0, 0.1, 0.2]))

    Nloops = 2
    optinds, opterrs = A.optparamfinder(labels, values, Nloops)
    print(optinds, opterrs)
    return A.models[0]

def xgblinearopter():
    model = XGBoost(num_round = 10)
    model.paramchanger('booster', 'gblinear')
    models = [model]
    loader = pulsardat()
    A = analyze(models, loader)

    labels = ['lambda',
              'alpha',
              'top_k'
             ]

    values = [np.array([0, 0.01, 0.1, 1]),
              np.array([0, 0.01, 0.1, 1]),
              np.array([0, 7, 6, 5])
             ]
    Nloops = 2
    optinds, opterrs = A.optparamfinder(labels, values, Nloops)
    print(optinds, opterrs)
    return A.models[0]

def NNopter():
    model = NNmodel()
    model.paramchanger('booster', 'gblinear')
    models = [model]
    loader = pulsardat()
    A = analyze(models, loader)

    labels = ['layers',
              'epochs',
              'batch_size'
             ]

    values = [[[128,2], [64,2], [32,2]],
              np.array([5,10,15,20]),
              np.array([8,16,32,64])            #from 8,16,32,64, 64 was selected
             ]

    Nloops = 2
    optinds, opterrs = A.optparamfinder(labels, values, Nloops)
    print(optinds, opterrs)
    return A.models[0]

def optmodelcomp():
    loader = pulsardat()
    model1 = xgbtreeopter('dart')
    model2 = xgbtreeopter('gbtree')
    model3 = xgblinearopter()
    model4 = LogReg()
    model5 = NNmodel()
    model5.paramchanger('layers', [64,2])           #selected values from NNopter
    model5.paramchanger('batch_size', 64)
    models = [model1, model2, model3, model4, model5]
    A = analyze(models, loader)
    A.traintestpred('ok')
    A.plot_PR()

def multimodelcomp():
    loader = pulsardat()
    model1 = XGBoost(num_round = 10)
    model4 = XGBoost(num_round = 10)
    #model4.paramchanger('objective', 'reg:squarederror')
    model4.paramchanger('booster', 'gblinear')
    model5 = XGBoost(num_round = 10)
    model5.paramchanger('booster', 'gbtree')
    model2 = NNmodel()
    model3 = NNxgb(10)
    model6 = LogReg()
    models = [model1, model2, model3, model4, model5,model6]
    A = analyze(models, loader)
    A.traintestpred("ok")
    A.plot_PR()

if __name__ == "__main__":
    optmodelcomp()
    #ok = NNopter()
