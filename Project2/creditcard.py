import numpy as np
import importlib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
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

class credlog(Logistic):
    def __init__(self):
        pass

class credNN(FFNN):
    pass

class ccdata:
    def __init__(self, adjust = True, NN = True):
        self.df = pd.read_excel('ccdefaults.xls', skiprows = [1])
        if adjust:
            self.adjust()
        if NN:
            self.type = "NN"
        else:
            self.type = "logreg"

    def adjust(self):
        """
        Removes rows containing data not described in the credit card data set.
        May also for instance set genders 1 and 2, to -1 and 1 instead.
        """
        df = self.df
        Ts = np.ones((23,len(df)))
        Ts[1] = (df['X2'].values == 1) + (df['X2'].values == 2)
        Ts[2] = (df['X3'].values == 1) + (df['X3'].values == 2)\
            +(df['X3'].values == 3) + (df['X2'].values == 4)
        Ts[3] = (df['X4'].values == 1) + (df['X4'].values == 2)\
            +(df['X4'].values == 3)

        for i in range(6,12):
            t = np.zeros(len(df))>np.ones(len(df))
            for j in range(-2,10):
                t+=df['X%i'%i].values == j
            Ts[i-1] = t
        T = np.prod(Ts, axis=0)
        self.df = df[T.astype(bool)]
        self.df = self.df.sample(frac = 1.0)

    def corrplot(self):
        df = self.df
        vars = df.keys()[1:]
        dat = df[vars].values
        corr = np.corrcoef(dat.T).round(2)
        sns.heatmap(data = corr, annot = True, cmap = 'viridis', xticklabels = vars, yticklabels = vars)

    def __call__(self):
        df = self.df
        y = df['Y'].values
        X_vars = df.keys()[1:-1]
        npca = len(X_vars)#4
        #pca = PCA(n_components = npca)
        #Xnew = preprocessing.scale(pca.fit_transform(df[X_vars].values))
        Xnew = preprocessing.scale(df[X_vars].values)
        #X_vars = ['pca%i' for i in range(1,npca+1)]
        for i in range(npca):
            df[X_vars[i]] = Xnew[:,i]
        X_vars = ['X1', 'X2', 'X3', 'X6', 'X7']
        #df = preprocessing.scale(df)

        if self.type=="NN":
            y_vars = ['def', 'nodef']
            df['def'] = (y==1)*1
            df['nodef'] = (y==0)*1

            return df, X_vars, y_vars
        elif self.type == "logreg":
            y_vars = ['Y']
            return df, X_vars, y_vars

def save_results_latex(self,filename,results,format_types):
    """
    Adds result to filename, stored in latex table format
    Results should be a list of numbers.
    format_types should be string like "%.3f" that specifies how each
    column sould be formatted
    """
    file = open(filename,'a')
    string = ''
    for i,number in enumerate(results):
        string += "%s&"%(format_types[i])%(number)
    string = string[:-1]
    string += "\\\ \n \hline \n"
    file.write(string)
    file.close()

if __name__ == "__main__":
    """
    ccd = ccdata(NN = True)
    N1 = credNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), loader = ccd)
    N1.train(200)
    N1.feedforward()
    print(N1.trainpredict(), N1.testpredict())
    """
    loader = ccdata(NN = False)
    LogAnalyze = ModelAnalysis(Logistic(), loader)
    tn, fp, fn, tp = LogAnalyze.kfolderr(Cmat(),ks = 5, frac = 1.0,N = 100,eta = 0.2,M = 128)
    x_data, y_data = LogAnalyze.ROCcurve(N = 100, eta = 0.2, M = 128)
    loader.type = "NN"
    NNmodel = FFNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), Xfeatures = 5, yfeatures = 2)
    NNAnalyze = ModelAnalysis(NNmodel, loader)
    n_epochs = 100; batches = 10
    NNAnalyze.kfolderr(FalseRate(), 5, 1.0, n_epochs, batches)
    #NNAnalyze.kfolderr(A)
