import numpy as np
import importlib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import LogisticRegression as L
importlib.reload(L)
from LogisticRegression import Logistic
from NeuralNet import FFNN
import Functions as F
importlib.reload(F)
from Functions import *
import Analyze as a
importlib.reload(a)
from Analyze import ModelAnalysis
from sklearn import preprocessing

class cancerdata:
    def __init__(self, adjust = True, NN = True):
        cancer = datasets.load_breast_cancer()
        self.df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
        self.df['target'] = cancer.target
        self.df = self.df.sample(frac=1.0)
        if adjust:
            self.adjust()
        if NN:
            self.type = "NN"
        else:
            self.type = "logreg"

        self.Xf = 4
        self.yf = 2

    def adjust(self):
        pass

    def corrplot(self):
        df = self.df
        vars = df.keys()
        dat = df[vars].values
        corr = np.corrcoef(dat.T).round(2)
        sns.heatmap(data = corr, annot = True, cmap = 'viridis', xticklabels = vars, yticklabels = vars)

    def __call__(self):
        df = self.df
        y = df['target'].values
        X_vars = df.keys()[:-1]
        npca = 4#len(X_vars)#4
        pca = PCA(n_components = npca)
        Xnew = preprocessing.scale(pca.fit_transform(df[X_vars].values))
        #Xnew = preprocessing.scale(df[X_vars].values)
        X_vars = ['pca%i'%i for i in range(1,npca+1)]
        for i in range(npca):
            df[X_vars[i]] = Xnew[:,i]
        #df = preprocessing.scale(df)


        if self.type=="NN":
            y_vars = ['c', 'noc']
            df['c'] = (y==1)*1
            df['noc'] = (y==0)*1

            return df, X_vars, y_vars
        elif self.type == "logreg":
            y_vars = ['target']
            return df, X_vars, y_vars

if __name__ == "__main__":
    loader = cancerdata(NN = True)
    # LogAnalyze = ModelAnalysis(Logistic(), loader)
    # wat = LogAnalyze.kfolderr(Accuracy(), ks = np.arange(2,6), frac = 1.0, N = 1000, eta = 0.2, M = 20)

    #loader.type = "NN"
    NNmodel = FFNN(hlayers = [30], activation = ReLU(0.01), outactivation = Softmax(),
                     cost = CrossEntropy(), Xfeatures = loader.Xf, yfeatures = loader.yf, eta = 0.1)
    # NNmodel.doreset = True
    # NNAnalyze = ModelAnalysis(NNmodel, loader)
    n_epochs = 100; batches = 10
    # NNAnalyze.kfolderr(FalseRate(), 5, 1.0, n_epochs, batches)
    df, Xv, yv = loader()
    dftrain, dftest = np.split(df, [int(0.8*len(df))])
    xtrain = dftrain[Xv].values; xtest = dftest[Xv].values
    ytrain = dftrain[yv].values; ytest = dftest[yv].values
    NNmodel.fit(xtrain, ytrain, n_epochs, batches)
    ypred = np.round(NNmodel.predict(xtest))
    y = dftest[yv].values
    Lmodel = Logistic()
    Lmodel.fit(xtrain, ytrain, n_epochs, 0.1, 10)
    Lpred = np.round(Lmodel.predict(xtest))
    F = FalseRate()
    print(F(Lpred, y))
    print(F(ypred, y))
    #NNAnalyze.kfolderr(A)
