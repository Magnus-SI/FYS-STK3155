import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split as sklsplit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.tri as mtri
from numpy.polynomial.polynomial import polyvander2d
import pandas as pd
import sys



class Ridge():
    def __init__(self,lambda_):
        """
        Parameters:
        lambda_ : Hyperparameter
        """
        self.lambda_ = lambda_

    def __call__(self,X,y):
        """
        Preforms Ridge regression, returns an array of estimates for the beta-vector

        Parameters:
        X : Design matrix
        y : Datapoints
        """
        lmb_matrix = np.identity(X.shape[1])*lambda_
        return np.linalg.inv(np.transpose(X)@X+lmb_matrix)@np.transpose(X)@y

class Lasso():
    def __init__(self,lambda_):
        self.lambda_ = lambda_

    def __call__(self):
        pass ## TODO: fill inn

def OLS(X,y):
    """
    OLS using our given formulas, note that this as of now does not work with
    polynomials of too high degree.
    """
    XtXinv=np.linalg.inv(np.einsum('ij,ik',X,X))
    return np.einsum('ij,kj,k',XtXinv,X,y)

def Ridge(X,y,lambda_):
    """
    Preforms Ridge regression, returns an array of estimates for the beta-vector

    Parametes:
    X : Design matrix
    y : Datapoints
    lambda_ : Hyperparameter
    """
    lmb_matrix = np.identity(X.shape[1])*lambda_
    return np.linalg.inv(np.transpose(X)@X+lmb_matrix)@np.transpose(X)@y

def OLS2(X,y):
    #OLS using scikit-learn
    lr=LinearRegression(fit_intercept=False)
    lr.fit(X,y)
    return lr.coef_

def R2(y,y_model):
    """
    Returns the R2 score for a dataset y and a set of predicted y values, y_model

    Parametes:
    y : datapoints
    y_model : points estimated by model
    """
    score = 1 - (y - y_model)**2 / (y - np.mean(y_model))
    return score

class idk:
    def __init__(self, seed=2):
        np.random.seed(seed)
        self.data=False         #data is not yet generated
        self.hasfit=False       #a fit has not yet been made
        self.compnoisy = True   #evaluate error compared to noisy data
        pass

    def FrankeFunction(self,x,y):
        #The franke function as given
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def gendat(self,N,noisefraq=0.05, Function=FrankeFunction, deg=(2,2)):
        """
        The data generation could, and probably should be changed to generate
        linspaced data as to allow for easier plotting, but also easier fitting.
        However, the code should be general enough that it would work anyways.

        N: Amount of data points generated
        noisefraq: fraction of data range in the y-direction as standard deviation
        """
        df = pd.DataFrame()
        x1,x2 = np.random.uniform(0,1,size=(2,N))
        y_exact = self.FrankeFunction(x1,x2)
        sigma = (np.max(y_exact)-np.min(y_exact))*noisefraq
        mu = 0
        y = y_exact + np.random.normal(mu, sigma, size=N)
        self.x1, self.x2, self.y_exact, self.y = x1, x2, y_exact, y
        self.N = N
        self.data=True      #checks that data is generated
        df['x1'] = x1
        df['x2'] = x2
        df['y_exact'] = y_exact
        df['y'] = y
        self.df=df              #dataframe of all data

        self.X = polyvander2d(x1,x2,deg)        #the polynomial coefficient matrix

    def kfoldsplit(self,k=5):
        """
        Splits data into k equally sized sets, 1 of which will be used for testing,
        the rest for training
        """
        df=self.df.sample(frac=1.0)     #ensures random order of data
        splitinds = len(df) * np.arange(1,k)/k  #indices to split at
        splitinds = splitinds.astype(int)
        dfsplit = np.split(df,splitinds)    #contains k dataframes, with the different sets of data
        return dfsplit

    def kfolderr(self,ks=np.arange(2,6), method=OLS):
        """
        Evaluetes the kfold error
        """
        counter=0
        MSE=0
        for k in ks:
            dfsplit = self.kfoldsplit(k)                        #split data
            for i in range(len(dfsplit)):
                dftrain = pd.concat(dfsplit[:i]+dfsplit[i+1:])  #training data
                self.fit(method,dftrain)        #fit with training data
                dftest = dfsplit[i]             #test data
                MSE += self.testeval(dftest)    #MSE on test data
                counter+=1
        return MSE/counter      #average mean square error


    def fit(self, method, df=None):
        """
        fits given pandas dataframe, and sets the coefficients self.beta
        method: method of fit. Would eventually be a choice between OLS, Ridge, Lasso
        df: if None, then fit all data
        (lambda): necessary when Ridge and Lasso is implemented
        """
        if df is None:
            df=self.df
        y  = df['y']
        inds = df.index
        self.beta = method(self.X[inds], y)
        #y_pred = self.predy(df)
        self.hasfit=True        #a fit has now been made
        #self.y_pred = y_pred

    def testeval(self,dftest):
        """
        Evaluates MSE for current beta fit, on a given set of test data
        dftest: pandas dataframe containing test data
        """
        if not self.hasfit:
            print("Error : ")
            sys.exit(1)
        inds = dftest.index
        y_pred = self.X[inds]@self.beta
        if self.compnoisy:
            y = dftest['y']
        else:
            y = dftest['y_exact']
        N = len(y)
        MSE = 1/N * np.sum((y_pred - y)**2)
        return MSE

    def degvnoiseerr(self,degs=np.arange(1,7),noises=np.logspace(-4,2,10)):
        """
        Compares MSEs of different degree polynomial fits, when exposed to different noises
        """
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        self.compnoisy=True         #whether to compare to actual data, or noisy data
        for deg in degs:
            MSEs=np.zeros(len(noises))
            for i,noise in enumerate(noises):
                self.gendat(self.N, noisefraq=noise, deg = (deg, deg))      #generate data
                MSEs[i]=self.kfolderr(method=OLS2)                          #evaluate k-fold error
            plt.plot(noises,MSEs, label="polydeg: %i"%deg)                  #plot error vs noise
        plt.xlabel("sigma noise fraction")
        plt.ylabel("MSE")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()


    def Error(self, test=False, usenoisy=True):
        """
        WARNING: currently outdated, use at your own caution

        Input:
        usenoisy: whether to compare the error of the predicted y
        to the noisy y or the actual y

        Returns the MSE, and later the R2 score
        """

        if usenoisy:
            y=self.y

        else:
            y=self.y_exact

        if test:
            y=self.ytest

        MSE=1/self.N * np.sum((self.y_pred-y)**2)
        return MSE

    def ErrorAnalysis(self, poldeg=(4,4), noises=np.logspace(-2,2,5)):
        """
        WARNING: currently outdated, but should still work, use degvnoiseerr instead

        Some basic example of error analysis that compares the MSE error as
        it changes due to noise, and how this is different when comparing
        to noisy, or not noisy data.
        """

        noises=np.array(noises)
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        MSEregs=np.zeros(len(noises))
        MSEacts=np.zeros(len(noises))
        for i, noise in enumerate(noises):
            self.gendat(self.N, noisefraq=noise, deg=poldeg)
            self.compnoisy=True
            MSEregs[i] = self.kfolderr(method = OLS2)
            self.compnoisy=False
            MSEacts[i] = self.kfolderr(method = OLS2)

        ax.plot(noises,MSEregs,label="Compared to noisy data")
        ax.plot(noises,MSEacts,label="Compared to actual data")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("MSE")
        plt.xlabel("sigma/max(y)")
        plt.title(poldeg)
        plt.legend()
        plt.show()
        return MSEregs,MSEacts


    def plot3D(self,usenoisy=True,approx=False):
        """
        WARNING: currently outdated, but should be updated to work at some point

        usenoisy: whether to plot the noisy data, or the actual data
        approx: whether to plot the approximated fit of the data

        Note that this method works in general, but if we end up using linspaced
        data in a meshgrid, a normal contourplot would be better
        """

        if usenoisy:
            y=self.y
        else:
            y=self.y_exact
        if not self.data:
            print("Generate data first")
            return
        triang=mtri.Triangulation(self.x1,self.x2)          #nevessary for unevenly spaced data
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1, projection='3d')
        #ax.plot_trisurf(triang,self.y,cmap='jet')
        ax.scatter(self.x1, self.x2, y, marker='.', s=10, c='green', alpha=0.5)
        if approx:
            if self.hasfit:
                ax.scatter(self.x1, self.x2, self.y_pred, marker='.', s=10, c='black', alpha=0.5)
            else:
                print("Fit data first")

        ax.view_init(elev=60, azim=-45)

        """
        ax=fig.gca(projection='3d')
        surf=ax.plot_surface(self.x1, self.x2, self.y_exact)
        ax.set_zlim(-0.1,1.4)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        """

        plt.show()
        pass

if __name__=="__main__":

    I = idk()
    I.gendat(500, noisefraq=0.001)
    ks=np.arange(2,6)
    MSE=I.kfolderr(ks)
    degs=np.arange(1,10)
    noises=np.logspace(-4,2,20)
    I.degvnoiseerr(degs,noises)
    """
    I=idk()
    I.gendat(500,noisefraq=0.001)
    I.fit(I.OLS,(5,5))
    I.plot3D(True, True)
    #print(I.Error())
    I.N=2000
    noises=np.logspace(-2,0,50)
    reg5,act5=I.ErrorAnalysis(poldeg=(5,5),noises=noises)
    reg8,act8=I.ErrorAnalysis(poldeg=(8,8),noises=noises)
    fig=plt.figure()
    plt.plot(noises,reg5, label="deg 5 vs. noise")
    plt.plot(noises,reg8, label="deg 8 vs. noise")
    plt.plot(noises,act5, label="deg 5 vs. actual")
    plt.plot(noises,act8, label="deg 8 vs. actual")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    """