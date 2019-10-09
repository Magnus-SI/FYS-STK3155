import importlib
import project1 as reloader
importlib.reload(reloader)

from project1 import Project1, OLS, MSE, OLS2
from imageio import imread
import pandas as pd
import numpy as np
from Ridge import Ridge, Ridgeskl
from Lasso import Lasso
import matplotlib.pyplot as plt


def MSE(data,model):
    N = data.size
    return np.sum((data-model)**2)/N

def OLS3(X,y):
    beta = np.linalg.pinv(X)@y
    return beta

class Terrain(Project1):
    def set_data(self,data,deg = (8,8),indices=False):
        """
        Set the data
        data: array of data
        deg: polynomial degree to create design matrix
        indices: if True, investigate parts of data
        """
        self.polydeg = deg
        df = pd.DataFrame()
        self.N = data.size
        self.Nx, self.Ny = data.shape

        x1, x2 = np.meshgrid(np.arange(self.Ny),np.arange(self.Nx))
        if indices:
            x1 = x1[0:300, 300:500]
            x2 = x2[0:300, 300:500]
            data = data[0:300, 300:500]
            self.Nx, self.Ny = data.shape
            self.N = data.size
        x1 = x1.flatten()
        x2 = x2.flatten()
        y = data.flatten()

        df['x1'] = x1
        df['x2'] = x2
        df['y'] = y
        self.df = df
        self.changepolydeg(polydeg = deg)
        self.noexact = True

    def plot_terrain(self):
        """
        plots the terrain data in a 2D plot
        needs no input
        """
        plt.figure()
        plt.title('Terrain over Norway 1')
        plt.imshow(self.df['y'].values.reshape(self.Nx,self.Ny), cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot_fit(self):
        """
        plots the fit, need no input, but the method fit has to be run before
        """
        plt.figure()
        plt.title('Fit')
        print((self.X@self.beta).shape)
        plt.imshow((self.X@self.beta).reshape(self.Nx,self.Ny), cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def fit_frac(self,method,frac):
        """
        fits a random sample of the dataset
        takes parameter frac : the fraction of the dataset that should be fitted
        """
        df = self.df.sample(frac=frac)
        self.fit(method,df = df)


if __name__ == '__main__':
    filename = "SRTM_data_Norway_1.tif"
    terrain_data = imread(filename)
    # terrain = Terrain()
    plt.imshow(terrain_data)
    plt.show()

    def Tinit(frac):
        "Initialized an instance of Terrain, and prints maximum and minimum terrain height"
        T = Terrain()
        T.set_data(terrain_data, deg = (5,5), indices=True)
        T.frac = frac
        yvals = T.df['y'].values
        print("Maximum height: %im\nMinimum height: %im"%(np.max(yvals), np.min(yvals)))
        return T
    # terrain.set_data(terrain_data,deg = (7,7),indices = True)
    #
    # terrain.plot_terrain()


    def terrainlambdacomplexanalysis(method=Ridge, frac = 0.2, saveplot = False):
        """
        Uses lambda_vs_complexity_error from project1.py to search for optimal lambda and complexity
        method: method to use in the search, if lambds is array with multiple elements,
        would be either Ridge or Lasso, if not, OLS should also work.
        frac: fraction of data to use, default is set to use indiced data

        """
        T = Terrain()
        T.set_data(terrain_data, deg = (5,5), indices = True)       #uses indiced data by default

        T.frac = frac
        T.cost ="R2"
        lambds = np.logspace(-11,-1,11)
        polydegs = np.arange(2,25)      #WARNING: do not go much higher than 10 with the LASSO method
        R = Ridge
        optdeg, optlambd, optR2, optMSE = T.lambda_vs_complexity_error(lambds, polydegs, method, noise = 0, terrain=True, saveplot = saveplot)
        T.changepolydeg((optdeg, optdeg))
        Mlambd = method(optlambd)
        T.fit(method(optlambd))     #optimal fit
        if not saveplot:
            T.plot_terrain()
            T.plot_fit()
        """
        NOTE: need some smart way of saving optdeg, optlambd, optR2 an appending to a table,
        which in turn can be read by latex. Different sets of values would correspond to
        OLS vs. Lasso vs. Ridge methods, along with different fractions of data and similar.
        """
    def multifracsave():
        "Saves data and plots from multiple fractions of a method to a txt-file in latex format"
        fracs = np.array([2e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1, 2e-1])
        for frac in fracs:
            terrainlambdacomplexanalysis(method = Ridgeskl, frac=frac, saveplot=True)

    def fitdeg(deg = 50, frac = 0.2, method =OLS3):
        """
        Fits terrain to a certain degree for plotting, also determines the related k-fold error
        """
        T = Terrain()
        T.set_data(terrain_data, deg = (deg, deg), indices = True)
        T.frac = frac
        T.fit(method)
        T.plot_fit()
        plt.title("%s fit, degree %i"%(method.__name__, deg))
        MSE = T.kfolderr(method = method)
        T.cost = "R2"
        R2 = T.kfolderr(method = method)
        print("*%s*\nMSE: %.4e\nRMSE: %.4f\nR2: %.4f\n"%(method.__name__, MSE, np.sqrt(MSE), R2))

    def OLSbestvsridgebest():
        print("Errors found using k-fold cross validation:")
        fitdeg(deg = 22, frac=0.2, method = Ridgeskl(1e-11))    #from tables in report
        fitdeg(deg = 30, frac=0.2, method=OLS3)                 #higher degree of polynomial OLS = better


    def multiOLS(degs = np.arange(2,51), frac = 0.2):
        """
        Compares errors of OLS for multiple complexities.
        Note that frac = 0.2 in effect means 20% training data, and 100% test data, so use only for estimate errors.
        Using k-fold cross validation would take too long for this function. Optimal polynomial degree
        with more exact error can instead be found by the fitdeg function.
        """
        T = Terrain()
        T.frac = frac
        T.set_data(terrain_data, deg = (2, 2), indices = True)
        method = OLS3
        MSEs = np.zeros(len(degs))
        R2s = np.zeros(len(degs))
        for i, deg in enumerate(degs):
            print(deg)
            T.changepolydeg((deg, deg))
            T.fit(OLS3)
            T.cost = "MSE"
            MSEs[i] = T.testeval(T.df)
            T.cost = "R2"
            R2s[i] = T.testeval(T.df)

        #T.plot_fit()
        #plt.title("%s fit, degree %i"%(method.__name__, deg))
        return MSEs, R2s
    # terrain.fit_frac(R,frac)
    # print("betas = ",terrain.beta)
    # print("MSE = ",MSE(terrain.df['y'],terrain.X@terrain.beta))
    #
    # terrain.plot_fit()
    # deg = np.arange(3,21)
    # MSE_kfold = np.zeros_like(deg)
    # for i,d in enumerate(deg):
    #     print(i)
    #     terrain.set_data(terrain_data,deg=(d,d),indices = True)
    #     err = terrain.kfolderr(method=OLS3)
    #     MSE_kfold[i] = err
    # plt.figure()
    # plt.plot(deg,MSE_kfold)
    # plt.show()
    """
    terrain.fit(OLS3)


    terrain.fit(R)
    print("betas (Ridge, lambda = %g)"%(_lambda), terrain.beta)
    print("betas (Ridge, lambda = %g)"%(_lambda), MSE(terrain.df['y'],terrain.X@terrain.beta))
    """
