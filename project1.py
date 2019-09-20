import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import sklearn.linear_model as skl
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
from Ridge import Ridge


def OLS(X,y):
    """
    OLS using our given formulas, note that this as of now does not work with
    polynomials of too high degree.
    """
    XtXinv = np.linalg.inv(np.einsum('ij,ik',X,X))
    return np.einsum('ij,kj,k',XtXinv,X,y)

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

def var_beta(beta,y,y_model):
    """
    Retruns an array with the standard deviation of each beta parameter

    Parametes:
    beta : estimated array of betas
    y : actual datapoints
    y_model : modeled y values
    """
    sigma_sqr = np.sum()

def FrankeFunction(x,y):
    #The franke function as given
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def functest(x,y):
    return x**3-x*y**2

class idk:
    def __init__(self, seed=2):
        np.random.seed(seed)
        self.data = False         #data is not yet generated
        self.hasfit = False       #a fit has not yet been made
        self.compnoisy = True   #evaluate error compared to noisy data
        pass



    def gendat(self,N,noisefraq=0.05, Function=FrankeFunction, deg=(2,2), randpoints = True):
        """
        The data generation could, and probably should be changed to generate
        linspaced data as to allow for easier plotting, but also easier fitting.
        However, the code should be general enough that it would work anyways.
        N: Amount of data points generated
        noisefraq: fraction of data range in the y-direction as standard deviation
        deg: degree of polynomial to generate initial design matrix
        randpoints: whether to generate randomly spaced, or evenly spaced data

        Returns:
        Changes state of the class to include a dataframe of data with added noise,
        as well as the design matrix
        """
        self.polydeg = deg
        df = pd.DataFrame()
        self.N = N
        if randpoints:
            x1,x2 = np.random.uniform(0,1,size=(2,N))
            y_exact = Function(x1,x2)
        else:
            n = int(np.sqrt(N))
            x1 = np.linspace(0, 1, n)
            x2 = np.linspace(0, 1, n)
            x1, x2 = np.meshgrid(x1,x2)
            x1 = x1.flatten()
            x2 = x2.flatten()
            y_exact = Function(x1,x2)
            self.N = n**2


        df['x1'] = x1
        df['x2'] = x2
        df['y_exact'] = y_exact

        self.df = df
        self.changenoise(noisefraq = noisefraq)
        self.changepolydeg(polydeg = deg)

    def changenoise(self, noisefraq):
        "Changes the noise of the current data"
        y_exact = self.df['y_exact']
        sigma = (np.max(y_exact)-np.min(y_exact))*noisefraq
        mu = 0
        self.df['y'] = y_exact + np.random.normal(mu, sigma, size=self.N)

    def changepolydeg(self, polydeg=(5,5)):
        "Changes the polynomial degree of the design matrix"
        self.polydeg = polydeg
        self.X = polyvander2d(self.df['x1'], self.df['x2'], polydeg)

    def kfoldsplit(self,k=5):
        """
        Splits data into k equally sized sets, 1 of which will be used for testing,
        the rest for training
        """
        df = self.df.sample(frac=1.0)     #ensures random order of data
        splitinds = len(df) * np.arange(1,k)/k  #indices to split at
        splitinds = splitinds.astype(int)
        dfsplit = np.split(df,splitinds)    #contains k dataframes, with the different sets of data
        return dfsplit

    def kfolderr(self,ks=np.arange(2,6), method=OLS):
        """
        Evaluetes the kfold error
        """
        counter = 0
        MSE = 0
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
            df = self.df
        y  = df['y']
        inds = df.index
        self.beta = method(self.X[inds], y)
        #y_pred = self.predy(df)
        self.hasfit = True        #a fit has now been made
        #self.y_pred = y_pred

    def testeval(self,dftest):
        """
        Evaluates MSE for current beta fit, on a given set of test data
        dftest: pandas dataframe containing test data
        """
        if not self.hasfit:
            print("Error : run fit before testeval")
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

    def MSEvlambda(self, lambds, method=Ridge(0), polydeg=(5,5), noises = np.logspace(-4,-2,2), avgnum=3):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        MSEs = np.zeros(len(lambds))
        self.gendat(self.N, noisefraq = noises[0], deg = polydeg)
        for noise in noises:
            for i,lambd in enumerate(lambds):
                method.lambda_ = lambd
                for j in range(avgnum):
                    self.changenoise(noise)
                    MSEs[i]+= self.kfolderr(method=method)
                MSEs[i] *= 1/avgnum
            plt.plot(lambds,MSEs, label="noise: %g" %noise)
            plt.xlabel("lambda")
            plt.ylabel("MSE")
            plt.xscale("log")
            plt.yscale("log")
        plt.legend()
        plt.show()

    def degvnoiseerr(self,degs=np.arange(1,7),noises=np.logspace(-4,2,10)):
        """
        Compares MSEs of different degree polynomial fits, when exposed to different noises
        """
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        for deg in degs:
            MSEs = np.zeros(len(noises))
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
            y = self.y

        else:
            y = self.y_exact

        if test:
            y = self.ytest

        MSE=1/self.N * np.sum((self.y_pred-y)**2)
        return MSE

    def Bootstrap(self,K,model):
        """
        Calculates the confidence interval of each beta parameter through the
        bootstrap method

        Parameters :
        K : Numbers of iterations of bootstrap
        model : A fitting function, must take arguments (X,y) can be call function of a class

        Returns :
        sigma_beta : array containing the standard deviation of the beta values
        """
        betas = np.zeros(shape = (K,len(self.X[0,:])))
        # Run fit method K times
        for i in range(K):
            self.fit(model,df = self.df.sample(frac = 1.0, replace = True))
            betas[i] += self.beta
        # Find average
        avg_beta = np.sum(betas,axis = 0)/K
        # Calculate variance
        sigma_beta_sqr = np.sum((betas-avg_beta)**2,axis = 0)/K
        # Take square root to find the standard deviation
        sigma_beta = np.sqrt(sigma_beta_sqr)
        return sigma_beta

    def ErrorAnalysis(self, poldeg=(4,4), noises=np.logspace(-2,2,5)):
        """
        WARNING: currently outdated, but should still work, use degvnoiseerr instead

        Some basic example of error analysis that compares the MSE error as
        it changes due to noise, and how this is different when comparing
        to noisy, or not noisy data.
        """

        noises = np.array(noises)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        MSEregs = np.zeros(len(noises))
        MSEacts = np.zeros(len(noises))
        for i, noise in enumerate(noises):
            self.gendat(self.N, noisefraq=noise, deg=poldeg)
            self.compnoisy=True
            MSEregs[i] = self.kfolderr(method = OLS2)
            self.compnoisy=False
            MSEacts[i] = self.kfolderr(method = OLS2)

        ax.plot(noises,MSEregs,label = "Compared to noisy data")
        ax.plot(noises,MSEacts,label = "Compared to actual data")
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
            y = self.y
        else:
            y = self.y_exact
        if not self.data:
            print("Generate data first")
            return
        triang = mtri.Triangulation(self.x1,self.x2)          #nevessary for unevenly spaced data
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
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

    ks = np.arange(2,6)
    MSE = I.kfolderr(ks)
    #degs = np.arange(1,10)
    #noises = np.logspace(-4,2,20)
    #I.degvnoiseerr(degs,noises)

    lambds = np.logspace(-8,-1,8)
    I.MSEvlambda(lambds)
    print(I.Bootstrap(1000,OLS))
    print(I.beta)

    lambds = np.logspace(-8,-1,50)
    #I.MSEvlambda(lambds)
    #I = idk()
    #I.gendat(21**2, noisefraq = 1e-3, Function = functest, deg = (2,2), randpoints = False)
    #I.compnoisy=False
    #I.MSEvlambda(lambds, method = Ridge(0),  polydeg = (2,2), noises = np.logspace(-4,-1, 4), avgnum = 15)

    """
    I = idk()
    I.gendat(500,noisefraq=0.001)
    I.fit(I.OLS,(5,5))
    I.plot3D(True, True)
    #print(I.Error())
    I.N = 2000
    noises = np.logspace(-2,0,50)
    reg5,act5 = I.ErrorAnalysis(poldeg=(5,5),noises=noises)
    reg8,act8 = I.ErrorAnalysis(poldeg=(8,8),noises=noises)
    fig = plt.figure()
    plt.plot(noises,reg5, label="deg 5 vs. noise")
    plt.plot(noises,reg8, label="deg 8 vs. noise")
    plt.plot(noises,act5, label="deg 5 vs. actual")
    plt.plot(noises,act8, label="deg 8 vs. actual")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    """
