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
                      'layers': [4,8,16,32,2],
                      'kernel_size': 32,
                      'activations': ['relu','relu','relu', 'relu', 'softmax'],
                      'optimizer': 'adam',
                      'loss': 'categorical_crossentropy',
                      'metrics': ['Precision', 'Recall', 'FalseNegatives'],
                      'epochs': 10,
                      'batch_size': 32,
                      'input_len': 3197
                     }
        self.initmodel()

    def initmodel(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(filters=self.param['layers'][0],\
                  kernel_size= self.param['kernel_size'],\
                  activation = self.param['activations'][0],\
                  input_shape=(self.param['input_len'],1)))
        model.add(tf.keras.layers.MaxPool1D())
        for i,l in enumerate(self.param['layers'][1:-1]):
            model.add(tf.keras.layers.Conv1D(filters=self.param['layers'][i],\
                      kernel_size= self.param['kernel_size'],\
                      activation = self.param['activations'][i]))
            model.add(tf.keras.layers.MaxPool1D())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.param['layers'][-1],self.param['activations'][-1]))

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

if __name__ == "__main__":
    from models import NNmodel
    loader = exodat()
    model1 = one_dim_CNN()
    model2 = NNmodel()
    model3 = XGBoost(10)
    models = [model3, model1, model2]
    A = analyze(models, loader)
    A.traintestpred("ok")