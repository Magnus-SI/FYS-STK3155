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
        """
        Combination of a FFNN and XGB
        """
        self.paramxgb = {'max_depth': 3,
                      'eta': 1,
                      'objective': 'binary:logistic',
                      'nthread': 8,
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
        """
        Initialize the NN
        """
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
        """
        Changes parameters
        label: label of parameter as found in the param dictionaries
        value: value to change to
        paramtype: change xgb parameter or NN parameter
        """
        if paramtype == 'xgb':
            self.paramxgb[label] = value
        else:
            self.paramNN[label] = value
            #if label not in ['epochs', 'batch_size']:
            self.initNN()

    def expanddim(self,y):
        """
        Dimensions must be expanded for NN
        """
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fitNN(self, X, y):
        """
        Fits the Neural network to the data
        X: design matrix
        y: target class
        """
        if len(y.shape)==1:
            y = self.expanddim(y)
        self.NN.fit(X, y,
                      epochs = self.paramNN['epochs'],
                      batch_size = self.paramNN['batch_size'],
                      verbose = False
                      )

    def fitxgb(self, X, y):
        """
        Uses xgb to fit the data after it has passed through the first
        hidden layer of the NN
        """
        X = self.interm.predict(X)
        dfit = xgb.DMatrix(X, label = y)
        self.bst = xgb.train(self.paramxgb, dfit, self.num_round_xgb)

    def fit(self, X,y):
        """
        Fits the NN, and then fits the xgb based on the NN fit.
        """
        self.fitNN(X,y)
        self.fitxgb(X,y)

    def predict(self, X):
        """
        """
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
        """
        Changes parameters in the self.param dictionary
        label: label of param to change
        value: value to change to
        """

        if label == 'num_round':
            self.num_round = value
        else:
            self.param[label] = value

    def fit(self,X,y):
        dfit = xgb.DMatrix(X, label = y)
        self.model = xgb.XGBClassifier(**self.param)
        self.bst = xgb.train(self.param, dfit, self.num_round)

    def predict(self, X):
        dpred = xgb.DMatrix(X)
        return self.bst.predict(dpred)

    def feature_importances(self):
        """
        Returns the feature importances as in the feature_importances_ method for XGBs sklearn interface
        """
        return self.model.feature_importances_

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
        """
        Initializes the NN based on parameters currently in the self.param dictionary
        """
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
        """
        Changes parameters in the self.param dictionary
        label: label of param to change
        value: value to change to
        """

        self.param[label] = value
        # if label not in ['epochs', 'batch_size']:
        #     self.initmodel()
        self.initmodel()

    def expanddim(self,y):
        """
        Expands 1d target class to 2d binary class.
        """
        ynew = np.zeros((len(y), 2))
        ynew[:, 0] = y
        ynew[:, 1] = 1-y
        return ynew

    def fit(self,X,y):
        if len(y.shape)==1:
            y = self.expanddim(y)
        self.model.fit(X, y,
                      epochs = self.param['epochs'],
                      batch_size = self.param['batch_size'],
                      verbose = False
                      )

    def predict(self, X):
        return self.model.predict(X)[:,0]

class analyze:
    def __init__(self, models, loader):
        """
        models: list of models, e.g. [NN,xgboost]
        loader: initialized class or function that returns dataframe and labels of x and y
        """
        self.models = models
        self.df, self.xlabels, self.ylabels = loader()
        self.N = len(self.df)
        self.traintestsimple(train_frac = 0.8)

    def traintestsplit(self, traininds, testinds):
        """
        Generates dftrain and dftest arrays based on indices traininds and testinds
        """
        self.dftrain = self.df.iloc[traininds]
        self.dftest = self.df.iloc[testinds]

    def traintestsimple(self, train_frac):
        """
        Performs a simple train test split with the given fraction of training data
        """
        inds = self.df.index.values
        np.random.shuffle(inds)
        traininds, testinds = np.split(inds, [int(train_frac*self.N)])
        self.traintestsplit(traininds, testinds)

    def traintestresample(self, train_frac, K = 5):
        """
        Performs K train-test splits with a train_frac training data.
        Used for resampling in the hyperparameter optimization.
        The training and test data are set as lists self.dftrains
        and self.dftests containing K dataframes.
        """
        self.dftrains = [None]*K
        self.dftests = [None]*K
        inds = self.df.index.values
        for k in range(K):
            np.random.shuffle(inds)
            traininds, testinds = np.split(inds, [int(train_frac*self.N)])
            self.dftrains[k] = self.df.iloc[traininds]
            self.dftests[k] = self.df.iloc[testinds]

    def traintestpred(self, cost, printscores = False):
        """
        Fits models in self.models on the current training data in self.dftrain,
        and evaluates their performance on current test data in self.dftest.

        cost: currently unused, as the area under the precision recdall curve
        is always used as the cost to determine the model performances
        printscores: True if all scores should be printed, typically False during
        hyperparameter otpimization.
        """
        Xtrain = self.dftrain[self.xlabels].values
        ytrain = self.dftrain[self.ylabels].values
        Xtest = self.dftest[self.xlabels].values
        ytest = self.dftest[self.ylabels].values
        scores = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            model.fit(Xtrain, ytrain)
            ypred = model.predict(Xtest)
            #print(ytest.shape, ypred.shape)
            precision, recall, thresholds = PRcurve(ytest, ypred)#metrics.precision_recall_curve(ytest, ypred)
            scores[i] = metrics.auc(precision, recall)
            #scores[i] = metrics.accuracy_score(ytest, np.round(ypred))
        if printscores:
            print(scores)
        return scores[0]    #score returned for hyperparmeter optimization.

    def analyze_kfold(self,Ks,curve_funcs):
        """
        Plots the PR (precision-recall) curve for all models, using the PRcurve_kfold method
        Also saves the resulting AUC measures.
        saves AUC for the curve to auc_results.txt and returns list of figures
        """
        figs = [plt.figure() for func in curve_funcs]
        for i,model in enumerate(self.models):
            name = type(model).__name__
            if name == "XGBoost":
                name = model.param['booster']
            x,ys,aucs = self.curve_kfold(Ks,model,curve_funcs)

            save_results_latex("auc_results.txt", [name]+aucs, ["%s"] + ["%.3f"]*len(aucs))
            for i,fig in enumerate(figs):
                plt.figure(fig.number)
                plt.plot(x,ys[i],label=name)

        for fig in figs:
            plt.figure(fig.number)
            plt.legend()
        return figs

    def optparamfinder(self, labels, values, Nloops, split = 'kfold'):
        """
        Finds the optimal parameters in a given search space, using a method
        described further in the report.

        labels: list of labels
        values: list of arrays containing corresponding values
        Nloops: loop over the process Nloops times
        split: kfold performs a kfoldsplit, otherwise split is a fraction of
               training data to use in the train-test split, where a resampling
               of the self.dftrains and self.dftests dataframes are performed.
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
                    #err_arr[k] = self.traintestpred("ok")
                    if split == 'kfold':
                        err_arr[k] = self.traintestkfold(np.arange(3,6))
                    else:
                        err = 0
                        for dftrain,dftest in zip(self.dftrains, self.dftests):
                            self.dftrain = dftrain
                            self.dftest = dftest
                            err += self.traintestpred("ok")
                        err_arr[k] = err/len(self.dftrains)

                optind = np.argmax(err_arr)     #index of optimal value of the parameter
                optinds[i,j] = optind           #store index
                model.paramchanger(labels[j], arr[optind])    #change to optimal value
            opterrs[i] = np.max(err_arr)        #error for the foudn optimal values this loop
        return optinds, opterrs

    def traintestkfold(self,ks):
        """
        Evalutes k-fold error for values ks of k.
        """
        err = 0
        c = 0
        for k in ks:
            dfsplit = self.kfoldsplit(k)
            for i in range(k):
                c+=1
                self.dftest = dfsplit[k-1-i]
                if i==0:
                    self.dftrain = pd.concat(dfsplit[:-1])
                elif i==k:
                    self.dftrain = pd.concat(dfsplit[1:])
                else:
                    self.dftrain = pd.concat(dfsplit[:k-1-i] + dfsplit[k-i:])
                err += self.traintestpred("ok")
        return err/c

    def kfoldsplit(self,k):
        """
        Splits data into k equally sized sets, 1 of which will be used for testing,
        the rest for training
        df: dataframe of data to split
        """
        df = self.df
        df = df.sample(frac=1.0)     #ensures random order of data
        splitinds = len(df) * np.arange(1,k)/k  #indices to split at
        splitinds = splitinds.astype(int)
        dfsplit = np.split(df,splitinds)    #contains a list of k dataframes, with the different sets of data
        return dfsplit

    def kfold_analysis(self,Ks,curve_funcs):
        """
        Returns a list of figure for all curves plotted
        curve_funcs is a list of functions taking target and prediction as argument
        each function should retur x, y and threshold values for the curve (x and y from 0 to 1)
        Ks is a list of all values for K to be used
        Stores k-fold AUC for all curves, as well as tpr,precision and fpr to the file 'results.txt'
        """


        figs = [plt.figure() for func in curve_funcs]

        for n,model in enumerate(self.models):
            ys = [np.zeros(10000) for curve in curve_funcs]
            x = np.linspace(0,1,10000)
            aucs = [0 for curve in curve_funcs]
            name = type(model).__name__
            tpr = 0; prec = 0; fpr = 0;

            if name == "XGBoost":
                name = model.param['booster']
            for K in Ks:
                dfsplit = self.kfoldsplit(K)

                for i in range(K):
                    dftest = dfsplit[K-1-i]
                    if i==0:
                        dftrain = pd.concat(dfsplit[:-1])
                    elif i==K:
                        dftrain = pd.concat(dfsplit[1:])
                    else:
                        dftrain = pd.concat(dfsplit[:K-1-i] + dfsplit[K-i:])
                    print(f"Test split {i+1} of {K}")

                    #model.initmodel()
                    model.fit(dftrain[self.xlabels].values, dftrain[self.ylabels].values)
                    target = dftest[self.ylabels].values
                    pred = model.predict(dftest[self.xlabels].values)

                    if len(pred.shape) == 2:
                        pred = pred[:,0]

                    for i,func in enumerate(curve_funcs):
                        xvals, yvals, thresholds = func(target, pred)
                        ys[i] += np.interp(x,xvals,yvals)
                        aucs[i] += metrics.auc(xvals, yvals)

                    tn, fp, fn, tp = metrics.confusion_matrix(target, np.round(pred)).ravel()

                    tpr += tp/(tp + fn)
                    prec += tp/(tp + fp)
                    fpr += fp/(fp+tn)

            for i in range(len(ys)):
                ys[i] /= np.sum(Ks)
                aucs[i] /= np.sum(Ks)
            tpr /= np.sum(Ks)
            prec /= np.sum(Ks)
            fpr /= np.sum(Ks)

            save_results_latex("results.txt", [name]+aucs+[tpr,prec,fpr], ["%s"] + ["%.3f"]*(len(aucs)+3))
            for i,fig in enumerate(figs):
                plt.figure(fig.number)
                plt.plot(x,ys[i],label=name)
        return figs

def xgbtreeopter(ttype = 'dart', split = 'kfold'):
    """
    Optimizes some set of hyperparameters for tree based xgb models.
    ttype: dart or gbtree
    split: kfold or float fraction of training data.
    """
    model = XGBoost(num_round = 10)
    model.paramchanger('booster', ttype)
    models = [model]
    loader = pulsardat()
    A = analyze(models, loader)
    if split == 'kfold':
        pass
    else:
        resamps = 25        #resamples
        A.traintestresample(train_frac = split, K = resamps)

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
              np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
              np.array([3, 5, 10, 15, 30, 50]),
              np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0]),
              np.array([0, 0.25, 0.5, 0.75, 1]),
              np.array([1, 1.25, 1.5, 1.75, 2]),
              np.array([0, 0.25, 0.5, 0.75, 1]),
              np.array([0.1, 0.5, 1, 2, 3, 5, 8]),
              ]

    if ttype == 'dart':
        labels.append('rate_drop')
        values.append(np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2]))

    Nloops = 5
    optinds, opterrs = A.optparamfinder(labels, values, Nloops, split)
    if split == 'kfold':
        return A.models[0]
    else:       #creates table as seen in the report
        optloop = np.argmax(opterrs)
        print(optinds, opterrs)
        inds = optinds[optloop]
        results = np.zeros(len(inds)+2)
        for i in range(len(inds)):
            results[i+1] = values[i][inds[i]]
        results[0] = split
        results[-1] = opterrs[optloop]
        save_results_latex('%s.txt'%ttype,results,format_types=['%s']+["%g"]*len(inds)+['%.3f'])
    #return A.models[0]

def xgblinearopter():
    """
    Optimizes some set of hyperparameters for the linear xgb model
    """
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

def NNopter(split = 'kfold'):
    """
    Optimizes a set of hyperparameters for the NN model
    spit: either kfold or a fraction of training data
    """
    model = NNmodel()
    #model.paramchanger('booster', 'gblinear')
    models = [model]
    loader = pulsardat()
    A = analyze(models, loader)

    if split == 'kfold':
        pass
    else:
        resamps = 5     #resamples
        A.traintestresample(train_frac = split, K = resamps)

    labels = ['layers',
              'epochs',
              'batch_size'
             ]

    values = [[[128,2], [64,2], [32,2], [16,2], [8,2]],
              np.array([5,10,15,20]),
              np.array([32,64,128,256])            #from 8,16,32,64, 64 was selected
             ]

    Nloops = 3
    optinds, opterrs = A.optparamfinder(labels, values, Nloops, split)
    print(optinds, opterrs)
    if split == 'kfold':
        return A.models[0]
    else:
        optloop = np.argmax(opterrs)
        print(optinds, opterrs)
        inds = optinds[optloop]
        results = np.zeros(len(inds)+2)
        for i in range(len(inds)):
            if i!=0:
                results[i+1] = values[i][inds[i]]
            else:
                results[i+1] = values[i][inds[i]][0]
        results[0] = split
        results[-1] = opterrs[optloop]
        save_results_latex('NN.txt',results,format_types=['%s']+["%g"]*len(inds)+['%.3f'])

def PRcurve(target, pred):
    """
    Function to make the PR curve return x,y,threshold, like the ROC curve
    """
    precision, recall, threshold = metrics.precision_recall_curve(target,pred)
    return recall[::-1], precision[::-1], threshold[::-1]


def optmodelcomp():
    """
    Compares performanceo of optimal models, including ROC and PR curves,
    as well as area under curves. Also plots an example decision tree from
    tree based xgb models, along with importance of the features.

    The optimization of the xgb models are performed in this function, while
    the NN model is previously optimized and the LogReg model has no hyperparameters.
    """
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
    Ks = [3,4,5]

    funcs = [PRcurve,metrics.roc_curve]
    figs = A.kfold_analysis(Ks,funcs)

    plt.figure(figs[0].number)
    plt.xlabel("Recall",fontsize=14)
    plt.ylabel("Precision",fontsize=14)
    plt.grid()
    plt.legend()
    plt.savefig("Auc_PR.png")

    plt.figure(figs[1].number)
    plt.xlabel("False positive ratio",fontsize=14)
    plt.ylabel("True positive ratio",fontsize=14)
    plt.grid()
    plt.legend()
    plt.savefig("Auc_ROC.png")


    plt.figure()
    A.models[1].model.fit(A.df[A.xlabels].values, A.df[A.ylabels].values)
    xgb.plot_tree(A.models[1].model)
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    plt.savefig("tree_plot.pdf")

    plt.figure()

    importances = A.models[1].feature_importances()
    plt.bar(np.arange(len(A.xlabels)),importances)
    plt.ylabel("Importance", fontsize = 14)
    plt.xlabel("Predictor number", fontsize = 14)
    plt.savefig("importance.png")

    plt.show()

    """
    A.traintestpred('ok')
    A.plot_PR()
    """

def trainfraccompdart():
    """
    Produces the table of the impact of training data size on hyperparameters for DART as seen in the report
    """
    fracs = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8])
    for frac in fracs:
        xgbtreeopter('dart', split =frac)

def trainfraccompNN():
    """
    Produces the table of the impact of training data size on hyperparameters for NN, that can
    be found under additional figures on the github page.
    """
    fracs = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8])
    for frac in fracs:
        print("newfrac: ", frac)
        NNopter(split =frac)

def save_results_latex(filename,results,format_types):
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
    pass
