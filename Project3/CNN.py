import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

class exodat:
    def __init__(self):
        dftest = pd.read_csv('exoTest.csv')
        dftrain = pd.read_csv('exoTrain.csv')

        self.xlabels = dftrain.keys()[1:]
        self.ylabels = dftrain.keys()[0]
        self.dftest = dftest; self.dftrain = dftrain

    def pre_process(self):
        x_test = self.dftest[self.xlabels].values
        x_train = self.dftrain[self.xlabels].values
        y_test = self.dftest[self.ylabels].values
        y_train = self.dftrain[self.ylabels].values

        scaler = preprocessing.StandardScaler()

        scaled_x_test = scaler.fit_transform(x_test.T).T
        scaled_x_train = scaler.fit_transform(x_train.T).T

        scaled_df_test = pd.DataFrame(scaled_x_test, columns=self.xlabels)
        scaled_df_train = pd.DataFrame(scaled_x_train, columns=self.xlabels)

        y_test -= 1; y_train -= 1

        scaled_df_test[self.ylabels] = y_test
        scaled_df_train[self.ylabels] = y_train

        self.dftest = scaled_df_test
        self.dftrain = scaled_df_train


    def __call__(self):
        self.pre_process()
        return self.dftrain, self.dftest, self.xlabels, self.ylabels

#Somewhat different, as the exoplanet data is already split into train and test
class analyze:
    def __init__(self, models, loader):
        """
        models: list of models, e.g. [NN,xgboost]
        """
        self.models = models
        self.dftrain, self.dftest, self.xlabels, self.ylabels = loader()

    def traintestpred(self, cost):
        Xtrain = self.dftrain[self.xlabels].values
        ytrain = self.dftrain[self.ylabels].values
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values
        scores = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            model.fit(Xtrain, ytrain)
            ypred = model.predict(Xtest)
            tn, fp, fn, tp = metrics.confusion_matrix(ytest, np.round(ypred)).ravel()
            print(f"Test results for model number {i}, ")
            print(f"TN : {tn}  FP : {fp},  FN : {fn},  TP : {tp}\n\n")
        return tp

    def plot_PR(self):
        """
        Plots the PR (precision-recall) curve for the model using the train data
        """
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values
        for model in self.models:
            ypred = model.predict(Xtest)
            if len(ypred.shape)==2:
                ypred=ypred[:,0]
            precision, recall, thresholds = metrics.precision_recall_curve(ytest, ypred)
            plt.plot(precision,recall)


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

class one_dim_CNN:
    def __init__(self):
        """
        Last layer will be a regular dense layer, so the dimension should match the data
        """
        self.param = {
                      'CNNfilters': [64],
                      'DenseLayers' : [512,2],
                      'kernel_size': 5,
                      'activations': ['relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': [tf.keras.metrics.AUC(curve='PR'), 'FalseNegatives'],
                      'epochs': 15,
                      'batch_size': 32,
                      'input_len': 3197
                     }
        self.initmodel()

    def initmodel(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(filters=self.param['CNNfilters'][0],\
                  kernel_size= self.param['kernel_size'],\
                  activation = self.param['activations'][0],\
                  input_shape=(self.param['input_len'],1)))
        model.add(tf.keras.layers.MaxPool1D())
        for fnum in self.param['CNNfilters'][1:]:
            model.add(tf.keras.layers.Conv1D(filters=fnum,\
                      kernel_size= self.param['kernel_size'],\
                      activation = self.param['activations'][0]))
            #model.add(tf.keras.layers.Conv1D(filters=fnum,\
            #          kernel_size= self.param['kernel_size'],\
            #          activation = self.param['activations'][0]))
            model.add(tf.keras.layers.MaxPool1D(pool_size=5,strides=2))

        model.add(tf.keras.layers.Flatten())

        for N in self.param['DenseLayers'][:-1]:
            model.add(tf.keras.layers.Dense(N,self.param['activations'][0]))
        model.add(tf.keras.layers.Dense(self.param['DenseLayers'][-1],self.param['activations'][-1]))



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

    def expanddim_X(self,X):
        xnew = np.zeros(list(X.shape)+ [1])
        xnew[:,:,0] = X
        return xnew

    def expanddim_y(self,y):
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fit(self,X,y):
        if len(y.shape)==1:
            y = self.expanddim_y(y)
        if len(X.shape)==2:
            X = self.expanddim_X(X)

        self.model.fit(X, y,
                      epochs = self.param['epochs'],
                      batch_size = self.param['batch_size']
                      )

    def predict(self, X):
        if len(X.shape)==2:
            X = self.expanddim_X(X)
        return self.model.predict(X)[:,0]

def xgboostopter1():
    model = XGBoost(10)
    models = [model]
    loader = exodat()
    A = analyze(models, loader)

    labels = ['max_depth', 'eta']
    values = [np.arange(1,5), np.array([0.5,1,1.5])]
    Nloops = 2

    return A.optparamfinder(labels, values, Nloops)

def CNNopter1():
    model = one_dim_CNN()
    models = [model]
    loader = exodat()
    A = analyze(models, loader)

    labels = ['CNNfilters', 'DenseLayers', 'kernel_size', 'batch_size']
    values = [[[64], [32], [16], [8]],
              [[2], [512,2], [256,2], [128,2], [512,64,2]],
              [3,4,5,6],
              np.array([8, 16,32,64])
             ]
    return A.optparamfinder(labels, values, 1)


if __name__ == "__main__":
    #from models import NNmodel
    #loader = exodat()
    #model1 = one_dim_CNN()
    #model2 = NNmodel()
    #model3 = XGBoost(10)
    #models = [model3, model1, model2]
    #A = analyze(models, loader)
    #A.traintestpred("ok")


    optinds, opterrs = xgboostopter1()
    optinds, opterrs = CNNopter1()
