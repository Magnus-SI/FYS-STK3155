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
            +(df['X3'].values == 3) #+ (df['X2'].values == 4)
        Ts[3] = (df['X4'].values == 1) + (df['X4'].values == 2)\
            +(df['X4'].values == 3)

        for i in range(6,12):
            t = np.zeros(len(df))>np.ones(len(df))
            for j in range(-2,10):
                t+=df['X%i'%i].values == j
            Ts[i-1] = t

        T = np.prod(Ts, axis=0)
        df = df[T.astype(bool)]
        df = df.sample(frac = 1.0)

        weights = np.array([0.4, 0.3, 0.15, 0.08, 0.05, 0.02])
        new1 = np.sum(df[["X%i"%i for i in range(6,12)]].values * weights, axis = 1)
        new2 = (df['X2'].values - 1)*2 - 1
        new3 = df['X3'].values - 2
        new4 = np.sum(df[["X%i"%i for i in range(18,24)]].values * weights, axis = 1)
        new4 = (new4 - np.mean(new4))/np.std(new4)
        new5 = df['X1'].values
        new5 = (new5-np.mean(new5))/np.std(new5)

        df['gender'] = new2
        df['education'] = new3
        df['payhist'] = new1
        df['prevpay'] = new4
        df['credit'] = new5
        self.df = df
        self.X_vars = ['credit', 'gender', 'education', 'payhist', 'prevpay']

    def corrplot(self):
        df = self.df
        vars = self.X_vars + ['Y']#df.keys()[1:]
        dat = df[vars].values
        corr = np.corrcoef(dat.T).round(2)
        sns.heatmap(data = corr, annot = True, cmap = 'viridis', xticklabels = vars, yticklabels = vars)

    def __call__(self):
        df = self.df
        y = df['Y'].values
        X_vars = self.X_vars#df.keys()[1:-1]
        #npca = 4#len(X_vars)#4
        #pca = PCA(n_components = npca)
        #Xnew = preprocessing.scale(pca.fit_transform(df[X_vars].values))
        #Xnew = preprocessing.scale(df[X_vars].values)
        # X_vars = ['pca%i'%i for i in range(1,npca+1)]
        # for i in range(npca):
        #     df[X_vars[i]] = Xnew[:,i]
        #X_vars = ['X1', 'X2', 'X3', 'X6', 'X7']
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
    N_epochs = 100
    Ltn, Lfp, Lfn, Ltp = LogAnalyze.kfolderr(Cmat(),ks = 5, frac = 1.0,N = N_epochs,eta = 0.2,M = 128)
    print(f"Results Logistic Regression (with {N_epochs} epochs):\
        \nTrue negative  : {Ltn}\
        \nFalse positive : {Lfp}\
        \nFalse negative : {Lfn}\
        \nTrue positie   : {Ltp}")
    Lx_data, Ly_data, LAUC= LogAnalyze.ROCcurve(N_run = 3, N = 1000, eta = 0.1, M = 128)

    loader.type = "NN"
    NNmodel = FFNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), Xfeatures = 5, yfeatures = 2)
    NNAnalyze = ModelAnalysis(NNmodel, loader)
    batch_size = 128
    batch_number = 100
    NNtn, NNfp, NNfn, NNtp = NNAnalyze.kfolderr(CmatNN(),ks = 5, frac = 1.0,n_epochs = N_epochs,eta = 0.2,batches = batch_number)
    print(f"Results Neural Network (with {N_epochs} epochs):\
        \nTrue negative  : {NNtn}\
        \nFalse positive : {NNfp}\
        \nFalse negative : {NNfn}\
        \nTrue positie   : {NNtp}")
    NNx_data, NNy_data, NNAUC = NNAnalyze.ROCcurve(N_run= 3,n_epochs = 1000, eta = 0.1, batches = batch_number)

    plt.figure()
    plt.plot(Lx_data,Ly_data,label="Logistic Regression")
    plt.plot(NNx_data,NNy_data,label="Neural Network")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()

    # n_epochs = 100
    #
    # NNAnalyze.kfolderr(FalseRate(), 5, 1.0, n_epochs, batches = batch_number, eta = 0.2)
    #NNAnalyze.kfolderr(A)
