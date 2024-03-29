import numpy as np
import pandas as pd
from sklearn import metrics

class ModelAnalysis:
    def __init__(self,model,loader):
        """
        model must have a 'fit'-function that takes X and y arrays
        Xsr and ystr are lists of strings such that df[X].values gives the X matrix
        and df[y].values gives the y-matrix
        """
        self.model = model
        self.df, self.Xstr, self.ystr = loader()

    def kfoldsplit(self,k, df):
        """
        Splits data into k equally sized sets, 1 of which will be used for testing,
        the rest for training
        df: dataframe of data to split
        """
        df = df.sample(frac=1.0)     #ensures random order of data
        splitinds = len(df) * np.arange(1,k)/k  #indices to split at
        splitinds = splitinds.astype(int)
        dfsplit = np.split(df,splitinds)    #contains a list of k dataframes, with the different sets of data
        return dfsplit

    def kfolderr(self,\
                costfunc,
                ks=np.arange(2,6),\
                frac = 1.0,\
                *args, **kwargs):
        """
        Evaluates the kfold error
        ks: the values of k to split in
        method: the method to evaluate the error on
        self.frac: can be smaller than 1 if dataset is large
        self.cost: the cost function to evaluate k-fold with
        *args and **kwargs are extra arguments required by model.fit
        """
        if type(ks) == int:
            ks = np.array([ks])
        counter = 0
        cost = 0
        df = self.df.sample(frac=frac)     #same set of data used for all k-splits
        if not self.model.hasfit:
            self.model.fit(self.df[self.Xstr].values, self.df[self.ystr].values, *args, **kwargs)
        for k in ks:
            dfsplit = self.kfoldsplit(k, df)                        #split data
            for i in range(len(dfsplit)):
                dftrain = pd.concat(dfsplit[:i]+dfsplit[i+1:])  #training data
                #fit with training data
                self.model.reset()
                self.model.fit(dftrain[self.Xstr].values, dftrain[self.ystr].values, *args, **kwargs)
                dftest = dfsplit[i]             #test data
                #cost on test data
                cost += costfunc(self.model.predict(dftest[self.Xstr].values), dftest[self.ystr].values)
                counter+=1
                print(f"k = {k}, run number {i+1}, cost = {cost/counter}")
        return cost/counter      #average of the cost function


    def trainerr(self, costfunc, *args):
        """
        Trains on, and evalutes error on the whole data set, or a fraction given by self.frac
        method: method of fitting, either a function or initialized class
        """
        df = self.df.sample(frac = self.frac)
        self.model.fit(df[self.Xstr].values,df[self.ystr].values, *args)
        cost = costfunc(self.model.predict(df[self.Xstr].values),df[self.ystr].values)
        return cost

    def ROCcurve(self,N_run,*args,**kwargs):
        """
        Returns FP rate, TP rate, and the area under the ROC curve
        """

        dfsplit = self.kfoldsplit(5,self.df)
        dftrain = pd.concat(dfsplit[:4])
        dftest = dfsplit[4]

        auc = 0
        fpr = np.linspace(0,1,1000)
        tpr = []
        for i in range(N_run):
            print(f"Test split {i+1} of {N_run}")
            self.model.reset()
            self.model.fit(dftrain[self.Xstr].values, dftrain[self.ystr].values, *args, **kwargs)
            target = dftest[self.ystr].values
            pred = self.model.predict(dftest[self.Xstr].values)

            if len(pred.shape) == 2:
                ind = np.argmax(pred.shape)
                if ind == 0:
                    pred = pred[:,0]
                else:
                    pred = pred[0,:]
            if len(target.shape) == 2:
                ind = np.argmax(target.shape)
                if ind == 0:
                    target = target[:,0]
                else:
                    target = target[0,:]
            values = metrics.roc_curve(target,pred)
            tpr.append(np.interp(fpr,values[0],values[1]))
            auc += metrics.auc(values[0], values[1])
            dfsplit = self.kfoldsplit(5,self.df)
            dftrain = pd.concat(dfsplit[:4])
            dftest = dfsplit[4]
        tpr = np.mean(tpr,axis = 0)
        auc /= N_run
        return fpr, tpr, auc
