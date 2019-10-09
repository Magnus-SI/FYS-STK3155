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
    XtXinv = np.linalg.inv(np.einsum('ij,ik',X,X))  #inverse of X^T X
    return np.einsum('ij,kj,k',XtXinv,X,y)          #beta coefficients

def OLS2(X,y):
    """
    OLS using scikit-learn
    """
    lr=LinearRegression(fit_intercept=False)
    lr.fit(X,y)
    return lr.coef_

def OLS3(X,y):
    """
    OLS using numpy pinv for the pseudo inverse
    """
    beta = np.linalg.pinv(X)@y
    return beta

class OLS3class:    #same as OLS3, used for general compatability with Ridge and Lasso
    def __init__(self,lambd):
        self.__name__ = "OLS"
    def __call__(self, X,y):
        """
        OLS using numpy pinv
        """
        beta = np.linalg.pinv(X)@y
        return beta


def FrankeFunction(x,y):
    "The Franke function"
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def functest(x,y):
    "Test function"
    return x**3-x*y**2

def sigma_beta_OLS(X, sigma):
    """
    Returns the standard deviation of the beta values calculated by OLS

    Paramaters :
    X : Design matrix (numpy array)
    sigma : the square root of the variance of the noise in the data
    """
    XTXinv_diag = np.diagonal(np.linalg.inv(np.transpose(X)@X))
    return np.sqrt(sigma*XTXinv_diag)

def sigma_beta_Ridge(X, sigma, _lambda):
    """
    Returns the standard deviation of the beta values calculated by Ridge

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
        self.noexact = False       #if True, data is loaded and not generated, so no y_exact will exist in self.df
        pass



    def gendat(self,N,noisefraq=0.05, Function=FrankeFunction, deg=(2,2), randpoints = True):
        """
        Generates data with a specifie3d function
        N: Amount of data points generated
        noisefraq: fraction of data range in the y-direction as standard deviation
        Function: function to generate data for
        deg: degree of polynomial, highest term of (x,y).
        randpoints: True if randomly distributed points, False if linspaced.
        If False, a countour plot is generated
        """
        self.polydeg = deg      #degree of current polynomial used in design matrix
        df = pd.DataFrame()
        self.N = N              #amount of data points
        if randpoints:
            x1,x2 = np.random.uniform(0,1,size=(2,N))
            y_exact = Function(x1,x2)
        else:
            n = int(np.sqrt(N))
            x1 = np.linspace(0, 1, n)
            x2 = np.linspace(0, 1, n)
            x1, x2 = np.meshgrid(x1,x2)
            plt.figure()
            plt.contourf(x1, x2, Function(x1,x2), levels = 15, cmap="viridis")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()          #shows contour of data
            x1 = x1.flatten()
            x2 = x2.flatten()
            y_exact = Function(x1,x2)
            self.N = n**2


        df['x1'] = x1               #indep. variable 1
        df['x2'] = x2               #indep. variable 2
        df['y_exact'] = y_exact     #exact data with no noise

        self.df = df
        if not randpoints:
            self.df = df.sample(frac=1.0)       #randomizes order if linspaced data
        self.changenoise(noisefraq = noisefraq) #changes noise to given fraction
        self.changepolydeg(polydeg = deg)       #changes polynomial degree

    def changenoise(self, noisefraq):
        """
        Changes the noise of the current data
        noisefraq: standard deviation of noise as a fraction of difference between
        maximum and minimum value
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
        Also normalizes the design matrix with respect to the largest element in each column
        """
        self.polydeg = polydeg
        X = polyvander2d(self.df['x1'], self.df['x2'], polydeg) #creates the design matrix
        norms = np.max(X,axis=0)    #maximum values of each column of the design matrix
        self.X = X/norms            #normalized by the maximum values of each column
        #self.norms = norms         #necessary if one wants unnormalized beta values

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

    def kfolderr(self,ks=np.arange(2,6), method=OLS):
        """
        Evaluates the kfold error
        ks: the values of k to split in
        method: the method to evaluate the error on
        self.frac: can be smaller than 1 if dataset is large
        self.cost: determines if the error is given as R2 or MSE
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


    def trainerr(self, method):
        """
        Trains on, and evalutes error on the whole data set, or a fraction given by self.frac
        method: method of fitting, either a function or initialized class
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
            df = self.df.sample(frac = self.frac)      #note that this randomizes order even if self.frac=1.0
        y  = df['y']
        inds = df.index
        self.beta = method(self.X[inds], y) #note that this beta is inverse normalized
        #y_pred = self.predy(df)
        self.hasfit = True        #a fit has now been made
        #self.y_pred = y_pred

    def testeval(self,dftest):
        """
        Evaluates MSE or R2 for current beta fit, on a given set of test data
        dftest: pandas dataframe containing test data
        self.cost: MSE or R2
        """
        if not self.hasfit:
            print("Error : run fit before testeval")
            sys.exit(1)
        inds = dftest.index
        y_pred = self.X[inds]@self.beta     #normalized X @ invnormalized beta -> unnormalized y
        if self.compnoisy:
            y = dftest['y']                 #compare to noisy data
        else:
            y = dftest['y_exact']           #compare to actual data
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

    def lambda_vs_complexity_error(self, lambds, polydegs, regtype, noise, showvals = True, new_plot = True, terrain=False, color = "#142cb1", saveplot=False):
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
        terrain: If True, do not plot training data, and do not use sigma in title
        color: color of the current line plot, not useful for 2d color plots.
        saveplot: if True, save plot instead of showing
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
                print(i,j)      #shows progress
                TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = regtype(lambd))
                TrainErrors[i,j] = self.trainerr(method = regtype(lambd))

        #Plotting the data
        if len(lambds) == 1:       #only tested for one value of lambda
            TestErrors.reshape(len(polydegs))
            TrainErrors.reshape(len(polydegs))

            if new_plot:
                plt.figure()
            plt.plot(polydegs, TestErrors, '%s'%color,  label="%s: Test"%(regtype.__name__))
            plt.plot(polydegs, TrainErrors, '%s'%color, linestyle='dashed', label="%s: Train"%(regtype.__name__))
            plt.xlabel("Polynomial degree")
            plt.ylabel(self.cost)
            plt.title(r"$\lambda = %g$, $\hat{\sigma} = %.1e$, %s datapoints"%(lambds[0], noise, self.N))
            plt.legend()
            #plt.savefig("test_lambs.pdf")
            plt.show()
            return

        if len(polydegs) == 1:       #only tested for one polynomial degree
            TeE = np.zeros(len(lambds))
            TrE = np.zeros(len(lambds))
            TeE[:] = TestErrors[0,:]
            TrE[:] = TrainErrors[0,:]
            if new_plot:
                plt.figure()
            plt.plot(np.log10(lambds), TeE, '%s'%color, label="%s: Test"%(regtype.__name__))
            plt.plot(np.log10(lambds), TrE, '%s'%color, linestyle='dashed', label="%s: Train"%(regtype.__name__))
            plt.xlabel(r"$log_{10}(\lambda)$")
            plt.ylabel(self.cost)
            plt.title(r"Polynomial degree %i, $\hat{\sigma} = %.1e$, %s datapoints"%(polydegs[0], noise, self.N))
            plt.legend()
            #plt.savefig("test_polydegs.pdf")
            plt.show()
            return

        if self.cost=="R2":
            vmin = 0.7; vmax = 1                #NOTE: change this if worse R2 scores
            optarg = np.argmax(TestErrors)      #index of optimal error
            optdegind = optarg//len(lambds)     #index of optimal degree
            optlambdind = optarg%len(lambds)    #index of optimal lambda
            optdeg = polydegs[optdegind]
            optlambd = lambds[optlambdind]
            optR2 = TestErrors[optdegind, optlambdind]
            self.cost = "MSE"
            self.changepolydeg((optdeg, optdeg))
            optMSE = self.kfolderr(ks = np.arange(2,6), method = regtype(optlambd))
            self.cost = "R2"
            print("Best R2: %.4f\nCorresponding MSE: %.2e\nOptimal degree: %i\nOptimal log10(lambda): %g"%(optR2, optMSE, optdeg, np.log10(optlambd)))
            if terrain:
                self.save_results_latex(filename = "terrain%s.txt"%(regtype.__name__), results = [int(self.N*self.frac), optR2, np.sqrt(optMSE), optdeg, np.log10(optlambd)], format_types = ['%i', '%.4f', '%.3e', '%i', '%.1f'])
            else:
                self.save_results_latex(filename = "franke%scompn%s.txt"%(regtype.__name__, self.compnoisy), results = [np.log10(noise), int(self.N*self.frac), optR2, np.sqrt(optMSE), optdeg, np.log10(optlambd)], format_types = ['%.1f', '%i', '%.4f', '%.3e', '%i', '%.1f'])

        elif self.cost=="MSE":
            vmin = False; vmax = False

        if terrain:
            plt.figure(figsize = (12,12))
            sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',xticklabels=np.log10(lambds), yticklabels=polydegs, vmin = vmin, vmax = vmax)
            plt.xlabel(r'$log_{10}(\lambda)$')
            plt.ylabel('Polynomial degree')
            plt.title('Terrain %s for %i data points, using %s'%(self.cost, int(self.N*self.frac), regtype.__name__))
            if saveplot:
                print("was here")
                plt.savefig("Terrainfigs/%s_%i.png"%(regtype.__name__, int(self.N*self.frac)))
                plt.close()
            else:
                plt.show()
            if self.cost == "R2":
                return optdeg, optlambd, optR2, optMSE

        f, axs = plt.subplots(2,1, figsize=(12,12))
        ax1, ax2 = axs
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.around(np.log10(lambds), 1), yticklabels=polydegs, vmin = vmin, vmax = vmax)
        ax1.set_xlabel(r'$log_{10}(\lambda)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title(r'%s Test Error, $\hat{\sigma} = %.1e, #datapoints = %i$'%(regtype.__name__, noise, int(self.N*self.frac)))
        h2=sns.heatmap(data=TrainErrors,annot=showvals,cmap='viridis',ax=ax2,xticklabels=np.around(np.log10(lambds), 1), yticklabels=polydegs, vmin = vmin, vmax = vmax)
        ax2.set_xlabel(r'$log_{10}(\lambda)$')
        ax2.set_ylabel('Polynomial degree')
        ax2.set_title(r'%s Train Error, $\hat{\sigma} = %.1e$'%(regtype.__name__, noise))
        if saveplot:
            plt.savefig("Frankefigs/%s_%.1f_%i_compn%s.png"%(regtype.__name__, np.log10(noise), int(self.N*self.frac), self.compnoisy))
            plt.close()
        else:
            plt.show()
        #return TestErrors, TrainErrors

    def degvnoiseerr(self, method, degs, noises, new_plot = True):
        """
        Compares error
        method: OLS/Ridge/Lasso, initialized with lambda
        degs: polynomial degrees to test for
        noises: noises to test fore
        new_plot: if False, allows for plotting multiple methods in the same figure
        """
        if new_plot:
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
        for deg in degs:
            self.changepolydeg((deg, deg))
            costs = np.zeros(len(noises))
            for i,noise in enumerate(noises):
                self.changenoise(noise)
                costs[i]=self.kfolderr(method=method)                          #evaluate k-fold error
            plt.plot(np.log10(noises),costs, label="%s: degree %i"%(method.__name__, deg))                  #plot error vs noise
        plt.xlabel(r"$log_{10}(\sigma)$")
        plt.ylabel(self.cost)
        #plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    def biasvar(self,K, model, polydegs):
        """
        Evaluates bias and variance for different polynomial degrees,
        using bootstrap to resample.
        K: Number of samples in bootstrap
        model: model to fit
        polydegs: array of polynomial degrees to compare.
        """
        split = int(0.8*self.N)             #80-20 train-test split
        MSEs = np.zeros(len(polydegs))
        biass = np.zeros(len(polydegs))
        variances = np.zeros(len(polydegs))
        dftrain, dftest = np.split(self.df, [split])    #performs the split
        testinds = dftest.index
        y = dftest['y'].values
        msebest = 1e10                      #temporary worst value, for saving the best value
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
        if self.cost=="MSE":
            plt.yscale("log")
        plt.xlabel("Degree of polynomial")
        plt.ylabel(self.cost)
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

    def plot3D(self,usenoisy=True,approx=False):
        """
        WARNING: not tested for a while, but should still work

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



if __name__=="__main__":
    def calculate_beta_variances(N,K,p,lambd,noisefraq = 1e-2):
        """
        Calculate the standard deviations for the beta paramaters
        from the theoretical expressions, and with the bootstrap method
        Uses theoretical expressions for OLS and rigde, and bootstrap on all
        methods. Prints results to terminal

        Parameters:
        N - number of datapoints
        K - nubmer of iterations of bootstrap
        p - polynomial degree to fit
        lambd - lambda parameter for ridge and lasso
        noisefraq - noise, relative to the difference between maximum and minimum of the dataset
        """
        P = Project1()
        P.gendat(N,noisefraq=noisefraq)
        R = Ridge(lambd)
        L = Lasso(lambd,tol = 1e-3)

        P.fit(OLS3)
        beta_OLS = P.beta

        P.fit(R)
        beta_ridge = P.beta

        P.fit(L)
        beta_lasso = P.beta

        sigma_beta_OLS_ = sigma_beta_OLS(P.X,P.sigma)
        sigma_beta_OLS_boot = P.Bootstrap(K,OLS3)
        sigma_beta_ridge_ = sigma_beta_Ridge(P.X,P.sigma,lambd)
        sigma_beta_ridge_boot = P.Bootstrap(K,R)
        #sigma_beta_lasso_boot = P.Bootstrap(K,L)

        OLS_results = np.zeros((len(beta_OLS),5))
        OLS_results[:,0] = beta_OLS
        OLS_results[:,1] = sigma_beta_OLS_
        OLS_results[:,2] = sigma_beta_OLS_boot
        OLS_results[:,3] = np.abs((sigma_beta_OLS_-sigma_beta_OLS_boot)/sigma_beta_OLS_)
        OLS_results[:,4] = (sigma_beta_OLS_+sigma_beta_OLS_boot)*0.5*1.96

        for i in range(len(beta_OLS)):
            P.save_results_latex("betas_OLS.txt",OLS_results[i,:],["%.2f"]*5)

        print("Results for OLS:\n","-"*20)
        print("Values for beta :\n")
        print(beta_OLS)
        print("sigma beta (theoretical):\n")
        print(sigma_beta_OLS_)
        print("sigma beta (Bootstrap):\n")
        print(sigma_beta_OLS_boot)
        print("Relative difference between bootstrap and theoretical:\n")
        print(np.abs((sigma_beta_OLS_-sigma_beta_OLS_boot)/sigma_beta_OLS_))

        print("\nResults for ridge\n","-"*20)
        print("Values for beta :\n")
        print(beta_ridge)
        print("sigma beta (theoretical):\n")
        print(sigma_beta_ridge_)
        print("sigma beta (Bootstrap):\n")
        print(sigma_beta_ridge_boot)
        print("Relative difference between bootstrap and theoretical:\n")
        print(np.abs((sigma_beta_ridge_-sigma_beta_ridge_boot)/sigma_beta_ridge_))

        """
        print("\nResults for LASSO\n","-"*20)
        print("Values for beta :\n")
        print(beta_lasso)
        print("sigma beta (Bootstrap):\n")
        print(sigma_beta_lasso_boot)
        """

    def methodsvsnoise(lambd = 1e-4):
        """
        Plots the error for different methods and different values of noise in the same plot
        """
        plt.close()
        P = Project1()
        P.gendat(500, noisefraq=1e-2)
        degs = np.arange(5,7)
        noises = np.logspace(-6,2,25)
        methods = (OLS3, Ridge(lambd))       #could also add Lasso, but takes a while.
        for method in methods:
            P.degvnoiseerr(method, degs, noises, new_plot=False)

    def methodsvscomplexity(lambd = 1e-6, N=200, noise = 1e-2, savefigs = False):
        """
        Plots training and test error as a function of complexity for different methods
        """
        I = Project1()
        I.gendat(N, noisefraq=1e-2)
        lambds = np.array([lambd])
        polydegs = np.arange(2,18)
        I.cost = "MSE"
        I.frac = 1.0
        I.compnoisy = False
        for regtype, color in zip([OLS3class, Ridge, Lasso], ["#cc1111", "#11cc11", "#1111cc"]):
            I.lambda_vs_complexity_error(lambds, polydegs, regtype, noise, new_plot=False, color = color)

        plt.yscale("log")
        if savefigs:
            plt.ylim(top=1e3, bottom = 1e-6)
            plt.savefig("biasvar/ORL_%i_%.1f_%.1f_compn%s.png"%(int(I.N*I.frac), np.log10(lambds[0]), np.log10(noise), I.compnoisy))
            plt.close()

    def multiOLSRidge():
        "Generates and saves multiplte train-test plots"
        lambds = np.array([1e-5])#np.logspace(-8,-4, 5)
        noises = np.array([1e-1])#np.logspace(-3,-1, 3)
        dpoints = np.array([100, 200, 500, 1000, 2000, 3000])#dpoints = np.array([100, 200, 500, 1000, 2000, 3000])
        for lambd in lambds:
            for noise in noises:
                for N in dpoints:
                    methodsvscomplexity(lambd, N, noise, savefigs = True)

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

    def lambdavcomplexityplots(noise=1e-1, N=400, saveplot = False):
        "Evaluates the different methods as a function of both polynomial degree and the hyperparameter lambda"
        I = Project1()
        I.gendat(N, noisefraq=1e-1)
        lambds = np.logspace(-11,-1,11)
        polydegs = np.arange(2,12)
        regtype = Ridge
        I.cost = "R2"
        I.frac = 1.0
        I.compnoisy=False
        I.lambda_vs_complexity_error(lambds, polydegs, regtype, noise, showvals=True, saveplot = saveplot)
        # I.cost = "MSE"
        # lambd = np.array([10**-5])
        # polydegs = np.arange(2,17)
        # I.lambda_vs_complexity_error(lambd, polydegs, regtype, noise)
        # lambd = np.array([10**-3])
        # polydegs = np.arange(2,25)
        # I.lambda_vs_complexity_error(lambd, polydegs, regtype, noise)

    def multilvcplots():
        """
        Saves optimal data and figures, for multiple noise values and datapoint counts
        of lambdavcomplexityplots.
        """
        noises = np.logspace(-3,-1, 3)
        datapoints = np.array([100, 200, 400, 1000])
        for noise in noises:
            for N in datapoints:
                lambdavcomplexityplots(noise, N, saveplot=True)

    def biasvarplots(resamps = 50):
        """
        Plots bias vs. variance
        """
        P = Project1()
        lambd = 1e-5
        method = Ridge(lambd)#OLS3#Ridge(lambd)    #OLS3
        polydegs = np.arange(2,30)
        P.gendat(1000, noisefraq=1e-2)
        P.biasvar(resamps,method,polydegs)        #vs noisy data
        plt.title(r"$\hat{\sigma} = 1e-2$, Ridge(1e-5) vs. noisy, 1000 datapoints")
        P.compnoisy=False
        P.biasvar(resamps,method,polydegs)        #vs actual data
        plt.title(r"$\hat{\sigma} = 1e-2$, Ridge(1e-5) vs. actual, 1000 datapoints")
