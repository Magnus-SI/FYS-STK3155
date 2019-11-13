import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LogisticRegression import Logistic
from NeuralNet import FFNN
from Functions import *
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
        #self.X_vars = df.keys()[1:-1]#self.X_vars[1:]
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

def save_results_latex(filename,results,format_types):
    """
    Adds result to filename, stored in latex table format
    results should be a list of printable results.
    format_types should be strings like "%.3f" that specifies how each column sould be formatted.
    These strings should be in a list, or another iterable structure
    """
    file = open(filename,'a')
    string = ''
    for i,number in enumerate(results):
        string += "%s&"%(format_types[i])%(number)
    string = string[:-1]
    string += "\\\ \n \hline \n"
    file.write(string)
    file.close()

def save_true_false_results(t,tn,fp,fn,tp,filename):
    print(f"\
        \nTrue negative  : {tn}\
        \nFalse positive : {fp}\
        \nFalse negative : {fn}\
        \nTrue positie   : {tp}")
    acc = (tp+tn)/(tn + fp + fn + tp)
    save_results_latex(filename,[t,tn, fp, fn, tp, acc],["%.2f"]+["%.1f"]*4 + ["%.3f"])

def Analyze_LogReg(loader,N_epochs):
    loader.type = 'logreg'
    Thresholds = [0.5]
    LogAnalyze = ModelAnalysis(Logistic(), loader)

    for t in Thresholds:
        print(f"Logistic Regression (with {N_epochs} epochs, threshold = {t}):")
        Ltn, Lfp, Lfn, Ltp = LogAnalyze.kfolderr(Cmat(t),ks = 5, frac = 1.0,N = N_epochs,eta = 0.2,M = 128)
        save_true_false_results(t,Ltn, Lfp, Lfn, Ltp,"LogRegResults.txt")

def plot_LogReg(loader,N_epochs):
    loader.type = 'logreg'
    LogAnalyze = ModelAnalysis(Logistic(), loader)
    LogAnalyze.model.npred = min(LogAnalyze.df[LogAnalyze.Xstr].shape)
    Lx_data, Ly_data, LAUC = LogAnalyze.ROCcurve(N_run = 10, N = N_epochs, eta = 0.1, M = 128)
    plt.plot(Lx_data, Ly_data, label="Logistic Regression")

def Analyze_NN(loader,N_epochs):
    loader.type = 'NN'
    Thresholds = [0.5]
    Xf = len(loader.X_vars)
    NNmodel = FFNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), Xfeatures = Xf, yfeatures = 2)
    NNAnalyze = ModelAnalysis(NNmodel, loader)
    trainsize = 0.8*len(NNAnalyze.df)
    batch_size = 128
    batch_number = int(trainsize/batch_size)
    for t in Thresholds:
        print(f"Neural network (with {N_epochs} epochs, threshold = {t}):")
        NNtn, NNfp, NNfn, NNtp = NNAnalyze.kfolderr(CmatNN(t),ks = 5, frac = 1.0,n_epochs = N_epochs,eta = 0.2,batches = batch_number)
        save_true_false_results(t,NNtn, NNfp, NNfn, NNtp,"NNResults.txt")

def plot_NN(loader,N_epochs,act_func):
    loader.type = 'NN'
    Xf = len(loader.X_vars)
    NNmodel = FFNN(hlayers = [30,15], activation = Sigmoid(), outactivation = Softmax(), cost = CrossEntropy(), Xfeatures = Xf, yfeatures = 2)
    NNAnalyze = ModelAnalysis(NNmodel, loader)
    trainsize = 0.8*len(NNAnalyze.df)
    batch_size = 128
    batch_number = int(trainsize/batch_size)
    for func in act_func:
        NNmodel.activation = func
        NNx_data, NNy_data, NNAUC = NNAnalyze.ROCcurve(N_run= 10,n_epochs = N_epochs, eta = 0.1, batches = batch_number)
        plt.plot(NNx_data,NNy_data,label=f"Neural Network with {type(NNmodel.activation).__name__}")


if __name__ == "__main__":
    """
    ccd = ccdata(NN = True)
    N1 = credNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), loader = ccd)
    N1.train(200)
    N1.feedforward()
    print(N1.trainpredict(), N1.testpredict())
    """

    Thresholds = [0.5]

    loader = ccdata(NN = False)
    N_epochs = 10

    Analyze_LogReg(loader, N_epochs)

    Analyze_NN(loader, N_epochs)

    act_func = [ReLU(0.01), ELU(0.01), Sigmoid()]

    plt.figure()

    plot_LogReg(loader,N_epochs)
    plot_NN(loader,N_epochs,act_func)

    plt.plot([0,1],[0,1],'k--', label = "Baseline")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.grid()
    plt.savefig("Classificationfigs/ROC_credit.pdf")
    plt.show()

    # n_epochs = 100
    #
    # NNAnalyze.kfolderr(FalseRate(), 5, 1.0, n_epochs, batches = batch_number, eta = 0.2)
    #NNAnalyze.kfolderr(A)
