import numpy as np
from sklearn.linear_model import LinearRegression
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
from Ridge import Ridge, Ridgeskl
from Lasso import Lasso
import seaborn as sns

def OLS(X,y):
    """
    OLS using our given formulas, note that this as of now does not work with
    polynomials of too high degree.
    """
    XtXinv = np.linalg.inv(np.einsum('ij,ik',X,X))
    return np.einsum('ij,kj,k',XtXinv,X,y)

def OLS2(X,y):
    """
    OLS using scikit-learn
    """
    lr=LinearRegression(fit_intercept=False)
    lr.fit(X,y)
    return lr.coef_

def OLS3(X,y):
    """
    OLS using numpy pinv
    """
    beta = np.linalg.pinv(X)@y
    return beta

class OLS3class:
    def __init__(self,lambd):
        pass
    def __call__(self, X,y):
        """
        OLS using numpy pinv
        """
        beta = np.linalg.pinv(X)@y
        return beta

def R2(y,y_model):
    """
    Returns the R2 score for a dataset y and a set of predicted y values, y_model

    Parametes:
    y : datapoints
    y_model : points estimated by model
    """
    score = 1 - (y - y_model)**2 / (y - np.mean(y_model))
    return score


def FrankeFunction(x,y):
    #The franke function as given
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def functest(x,y):
    return x**3-x*y**2

def var_beta_OLS(X, sigma):
    """
    Returns the variance of the beta values calculated by OLS

    Paramaters :
    X : Design matrix (numpy array)
    sigma : the square root of the variance of the noise in the data
    """
    XTXinv_diag = np.diagonal(np.linalg.inv(np.transpose(X)@X))
    return np.sqrt(sigma*XTXinv_diag)

def var_beta_Ridge(X, sigma, _lambda):
    """
    Returns the variance of the beta values calculated by Ridge

    Paramaters :
    X : Design matrix (numpy array)
    sigma : the square root of the variance of the noise in the data
    _lambda : hyperparameter used in the Ridge regression
    """
    lambda_mat = np.eye(X.shape[1])*_lambda
    XTX = np.transpose(X)@X
    return np.sqrt(np.diagonal(sigma*np.linalg.inv(XTX+lambda_mat)@XTX@np.transpose(np.linalg.inv(XTX + lambda_mat))))

class Project1:
    def __init__(self, seed=2):
        np.random.seed(seed)
        self.data = False         #data is not yet generated
        self.hasfit = False       #a fit has not yet been made
        self.compnoisy = True   #evaluate error compared to noisy data
        self.cost = "MSE"       #defines which cost function to use
        self.frac = 1.0         #fraction of the data to use
        self.noexact = False       #if True, data is loaded and not generated, so no y_exact will exist
        pass



    def gendat(self,N,noisefraq=0.05, Function=FrankeFunction, deg=(2,2), randpoints = True):
        """
        The data generation could, and probably should be changed to generate
        linspaced data as to allow for easier plotting, but also easier fitting.
        However, the code should be general enough that it would work anyways.
        N: Amount of data points generated
        noisefraq: fraction of data range in the y-direction as standard deviation
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
        if not randpoints:
            self.df = df.sample(frac=1.0)
        self.changenoise(noisefraq = noisefraq)
        self.changepolydeg(polydeg = deg)

    def changenoise(self, noisefraq):
        """
        Changes the noise of the current data
        """
        if self.noexact:
            return          #y_exact does not exist, prevents error
        y_exact = self.df['y_exact']
        sigma = (np.max(y_exact)-np.min(y_exact))*noisefraq
        self.sigma = sigma
        mu = 0
        self.df['y'] = y_exact + np.random.normal(mu, sigma, size=self.N)

    def changepolydeg(self, polydeg=(5,5)):
        """
        Changes the polynomial degree of the design matrix
        Also normalizes the design matrix
        """
        self.polydeg = polydeg
        X = polyvander2d(self.df['x1'], self.df['x2'], polydeg)
        norms = np.max(X,axis=0)
        self.X = X/norms    #normalized
        #self.norms = norms

    def kfoldsplit(self,k, df):
        """
        Splits data into k equally sized sets, 1 of which will be used for testing,
        the rest for training
        """
        df = df.sample(frac=1.0)     #ensures random order of data
        splitinds = len(df) * np.arange(1,k)/k  #indices to split at
        splitinds = splitinds.astype(int)
        dfsplit = np.split(df,splitinds)    #contains k dataframes, with the different sets of data
        return dfsplit

    def kfolderr(self,ks=np.arange(2,6), method=OLS):
        """
        Evaluates the kfold error
        ks:
        self.frac: can be smaller than 1 if dataset is large
        """
        counter = 0
        cost = 0
        df = self.df.sample(frac=self.frac)     #same set of data used for all k-splits
        for k in ks:
            dfsplit = self.kfoldsplit(k, df)                        #split data
            for i in range(len(dfsplit)):
                dftrain = pd.concat(dfsplit[:i]+dfsplit[i+1:])  #training data
                self.fit(method,dftrain)        #fit with training data
                dftest = dfsplit[i]             #test data
                cost += self.testeval(dftest)    #cost on test data
                counter+=1
        return cost/counter      #average mean square error

    def trainvtesterr(self):
        """
        Splits data into training and test data, fits on training, and evaluates
        error on both training and tests.
        This can be done for varying complexities.
        """
        pass

    def trainerr(self, method):
        """
        Trains on, and evalutes error on the whole data set
        """
        df = self.df.sample(frac = self.frac)
        self.fit(method, df)
        cost = self.testeval(df)
        return cost

    def fit(self, method, df=None):
        """
        fits given pandas dataframe, and sets the coefficients self.beta
        method: method of fit. Would eventually be a choice between OLS, Ridge, Lasso
        df: if None, then fit all data
        (lambda): necessary when Ridge and Lasso is implemented
        """
        if df is None:
            df = self.df.sample(frac = self.frac)      #note that this randomizes even if self.frac=1.0
        y  = df['y']
        inds = df.index
        self.beta = method(self.X[inds], y) #note that this beta is inverse normalized
        #y_pred = self.predy(df)
        self.hasfit = True        #a fit has now been made
        #self.y_pred = y_pred

    def testeval(self,dftest):
        """
        Evaluates MSE for current beta fit, on a given set of test data
        dftest: pandas dataframe containing test data
        self.cost: MSE or R2
        """
        if not self.hasfit:
            print("Error : run fit before testeval")
            sys.exit(1)
        inds = dftest.index
        y_pred = self.X[inds]@self.beta     #normalized X @ invnormalized beta -> unnormalized y
        if self.compnoisy:
            y = dftest['y']
        else:
            y = dftest['y_exact']
        N = len(y)
        if self.cost == "MSE":
            MSE = 1/N * np.sum((y_pred - y)**2)
            return MSE
        elif self.cost == "R2":
            score = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            return score
        else:
            print("Choose from MSE or R2 as a cost function")
            sys.exit(1)

    def lambda_vs_complexity_error(self, lambds, polydegs, regtype, noise, showvals = True, new_plot = True):
        """
        Generates a heat map comparing performance of the hyperparamater lambda
        for either Ridge or Lasso, with varying complexity. Checks performance both
        on test-data using k-fold error, and on training data.
        *Input*
        lambds: lambda values to evaluate
        polydegs: polynomial degrees to test for, corresponding to complexity
        regtype: Ridge or Lasso
        noise: noise to add to the data
        showvals: True if show values on colors, False if not
        new_plot: Used to compare multiple methods in the same plot.
        cost function: self.cost determines whether to use R2 or MSE for plotting

        Note that train errors and test errors do not use exactly the same training set
        because of the nature of the k-fold error evaluation. Results should still be similar
        to if they were however.
        """
        TestErrors = np.zeros((len(polydegs), len(lambds)))
        TrainErrors = np.zeros((len(polydegs), len(lambds)))
        self.changenoise(noise)
        for i,deg in enumerate(polydegs):
            self.changepolydeg((deg,deg))
            for j,lambd in enumerate(lambds):
                print(i,j)
                #might reset lambda value of regtype here, instead of initializing new class all the time
                #can't be done right now because of how the Lasso function is structured.
                TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = regtype(lambd))
                TrainErrors[i,j] = self.trainerr(method = regtype(lambd))

        #Plotting the data
        if len(lambds) == 1:       #only tested for one value of lambda
            TestErrors.reshape(len(polydegs))
            TrainErrors.reshape(len(polydegs))

            if new_plot:
                plt.figure()
            plt.plot(polydegs, TestErrors, label="%s: Test"%(regtype.__name__))
            plt.plot(polydegs, TrainErrors, label="%s: Train"%(regtype.__name__))
            plt.xlabel("Polynomial degree")
            plt.ylabel(self.cost)
            plt.title(r"$\lambda = %g$"%lambds[0])
            plt.legend()
            plt.show()
            return

        if len(polydegs) == 1:       #only tested for one polynomial degree
            TeE = np.zeros(len(lambds))
            TrE = np.zeros(len(lambds))
            TeE[:] = TestErrors[0,:]
            TrE[:] = TrainErrors[0,:]
            if new_plot:
                plt.figure()
            plt.plot(np.log10(lambds), TeE, label="%s: Test"%(regtype.__name__))
            plt.plot(np.log10(lambds), TrE, label="%s: Train"%(regtype.__name__))
            plt.xlabel(r"$log10(\lambda)$")
            plt.ylabel(self.cost)
            plt.title("Polynomial degree %i"%polydegs[0])
            plt.legend()
            plt.show()
            return

        if self.cost=="R2":
            vmin = 0; vmax = 1
        elif self.cost=="MSE":
            vmin = False; vmax = False
        f, axs = plt.subplots(2,1, figsize=(12,12))
        ax1, ax2 = axs
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.log10(lambds), yticklabels=polydegs, vmin = vmin, vmax = vmax)
        ax1.set_xlabel(r'$log10(\lambda)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title('Test Error')
        h2=sns.heatmap(data=TrainErrors,annot=showvals,cmap='viridis',ax=ax2,xticklabels=np.log10(lambds), yticklabels=polydegs, vmin = vmin, vmax = vmax)
        ax2.set_xlabel(r'$log10(\lambda)$')
        ax2.set_ylabel('Polynomial degree')
        ax2.set_title('Train Error')
        plt.show()
        #return TestErrors, TrainErrors


    def MSEvlambda(self, lambds, method=Ridge(0), polydeg=(5,5), noises = np.logspace(-4,-2,2), avgnum=3):
        fig = plt.figure()
        plt.xlabel("Polynomial degree")
        plt.ylabel("Error")
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

    def biasvar(self,K, model, polydegs):
        split = int(0.8*self.N)
        MSEs = np.zeros(len(polydegs))
        biass = np.zeros(len(polydegs))
        variances = np.zeros(len(polydegs))
        dftrain, dftest = np.split(self.df, [split])
        testinds = dftest.index
        y = dftest['y'].values
        msebest = 1e10
        if not self.compnoisy:
            y = dftest['y_exact'].values
        for j,polydeg in enumerate(polydegs):
            self.changepolydeg((polydeg, polydeg))
            ypreds = np.zeros((len(dftest), K))
            for i in range(K):
                df = dftrain.sample(frac=1.0, replace=True)
                #df = dftrain
                self.fit(model, df)
                ypreds[:,i] = self.X[testinds]@self.beta
            MSEs[j] = np.mean(np.mean((y-ypreds.transpose())**2, axis=0))
            biass[j] = np.mean((y-np.mean(ypreds,axis=1))**2)
            variances[j] = np.mean(np.var(ypreds, axis=1))
            if MSEs[j]<msebest:     #allows for easy plotting of best fit
                msebest = MSEs[j]
                betopt = self.beta
                Xopt = self.X

            #print(MSE, bias, variance, self.sigma**2)
        self.beta = betopt
        self.X = Xopt

        plt.figure()
        plt.plot(polydegs, MSEs, label="MSE")
        plt.plot(polydegs, biass, label="bias")
        plt.plot(polydegs, variances, label="variance")
        plt.plot(polydegs, biass+variances,'--', label="bias+var")
        plt.legend()
        plt.yscale("log")
        plt.xlabel("Degree of polynomial")
        plt.ylabel("Errors")
        plt.show()

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
            betas[i] = self.beta
        # Find average
        avg_beta = np.sum(betas,axis = 0)/K
        # Calculate variance
        sigma_beta_sqr = np.sum((betas-avg_beta)**2,axis = 0)/K
        # Take square root to find the standard deviation
        sigma_beta = np.sqrt(sigma_beta_sqr)
        print(sigma_beta.shape)
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
            y = self.df['y']
        else:
            y = self.df['y_exact']
        #if not self.data:
        #    print("Generate data first")
        #    return
        x1 = self.df['x1']; x2 = self.df['x2']
        triang = mtri.Triangulation(x1,x2)          #nevessary for unevenly spaced data
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        #ax.plot_trisurf(triang,self.y,cmap='jet')
        ax.scatter(x1, x2, y, marker='.', s=10, c='green', alpha=0.5)
        if approx:
            ax.scatter(x1, x2, self.X@self.beta, marker='.', s=10, c='black', alpha=0.5)

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

    def methodsvscomplexity():
        I = Project1()
        I.gendat(200, noisefraq=1e-2)
        lambds = np.array([10**-6])
        polydegs = np.arange(2,18)
        I.cost = "MSE"
        I.frac = 1.0
        noise = 1e-2
        for regtype in [OLS3class, Ridge, Lasso]:
            I.lambda_vs_complexity_error(lambds, polydegs, regtype, noise, new_plot=False)
        plt.yscale("log")

    def methodsvlambda():
        P = Project1()
        P.gendat(200, noisefraq=1e-2)
        lambds = np.logspace(-13,-1,13)
        polydegs = np.array([10])
        P.cost = "MSE"
        P.frac = 1.0
        noise = 1e-2
        for regtype in [OLS3class, Ridge, Lasso]:
            P.lambda_vs_complexity_error(lambds, polydegs, regtype, noise, new_plot=False)
        plt.yscale("log")

    def lambdavcomplexityplots():
        I = Project1()
        I.gendat(200, noisefraq=1e-4)
        lambds = np.logspace(-9,-1,17)
        polydegs = np.arange(2,14)
        regtype = Ridge
        noise = 1e-2
        I.cost = "R2"
        I.frac = 1.0
        #I.compnoisy=False
        I.lambda_vs_complexity_error(lambds, polydegs, regtype, noise, showvals=True)
        I.cost = "MSE"
        lambd = np.array([10**-5])
        polydegs = np.arange(2,17)
        I.lambda_vs_complexity_error(lambd, polydegs, regtype, noise)
        lambd = np.array([10**-3])
        polydegs = np.arange(2,25)
        I.lambda_vs_complexity_error(lambd, polydegs, regtype, noise)

    def biasvarplots(resamps = 50):
        P = Project1()
        polydegs = np.arange(2,25)
        P.gendat(5000, noisefraq=1e-2)
        P.biasvar(resamps,OLS3,polydegs)        #vs noisy data
        P.compnoisy=False
        P.biasvar(resamps,OLS3,polydegs)        #vs actual data

    #I.gendat(2000, noisefraq=0.001)
    #I.biasvar(20,OLS3,np.arange(1,20))

    # I.gendat(5000, noisefraq=0.001)
    # I.biasvar(20, OLS3, np.arange(1,15))

    # I.changepolydeg(polydeg = (5,5))
    # sigma_beta_Boot_OLS = I.Bootstrap(1000,OLS3)

    # #print("betas = ",I.beta)
    #
    # _lambda = 0.01
    # R = Ridge(_lambda)
    #
    # sigma_beta_Boot_Ridge = I.Bootstrap(1000,R)
    # sigma_beta_theoretical_OLS = var_beta_OLS(I.X, I.sigma)
    # sigma_beta_theoretical_Ridge = var_beta_Ridge(I.X, I.sigma,_lambda)
    #
    # print("beta variances theoretical (OLS) = \n", sigma_beta_theoretical_OLS)
    # print("beta variances bootstrap (OLS) = \n", sigma_beta_Boot_OLS)
    # print("relative difference (OLS) = \n", np.abs(sigma_beta_theoretical_OLS-sigma_beta_Boot_OLS)/np.abs(sigma_beta_theoretical_OLS))
    #
    # print("beta variances theoretical (Ridge) = \n", sigma_beta_theoretical_Ridge)
    # print("beta variances bootstrap (Ridge) = \n", sigma_beta_Boot_Ridge)
    #
    # formatter = lambda x:'%.2f'%(x)
    # print("relative difference (Ridge) = ")
    # Ridge_diff = np.abs(sigma_beta_theoretical_Ridge-sigma_beta_Boot_Ridge)/np.abs(sigma_beta_theoretical_Ridge)
    # print(Ridge_diff)


    #ks = np.arange(2,6)
    #MSE = I.kfolderr(ks)
    #degs = np.arange(1,10)
    #noises = np.logspace(-4,2,20)
    #I.degvnoiseerr(degs,noises)

    #lambds = np.logspace(-8,-1,8)
    #I.MSEvlambda(lambds)
    #print(I.Bootstrap(1000,OLS))
    #print(I.beta)

    #lambds = np.logspace(-8,-1,50)
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
