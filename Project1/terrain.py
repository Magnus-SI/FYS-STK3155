from project1 import idk, OLS, MSE, OLS2
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

class Terrain(idk):
    def set_data(self,data,deg = (8,8),indices=False):
        """
        Set the data
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
        x1 = x1.flatten()
        x2 = x2.flatten()
        y = data.flatten()

        df['x1'] = x1
        df['x2'] = x2
        df['y'] = y
        self.df = df
        self.changepolydeg(polydeg = deg)

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
    terrain = Terrain()
    plt.imshow(terrain_data)
    plt.show()
    terrain.set_data(terrain_data,deg = (30,30),indices = True)

    terrain.plot_terrain()

    _lambda = 0.001
    frac = 0.8
    R = Ridge(_lambda)
    L = Lasso(_lambda)
    terrain.fit_frac(R,frac)
    print("betas = ",terrain.beta)
    print("MSE = ",MSE(terrain.df['y'],terrain.X@terrain.beta))

    terrain.plot_fit()
    """
    terrain.fit(OLS3)


    terrain.fit(R)
    print("betas (Ridge, lambda = %g)"%(_lambda), terrain.beta)
    print("betas (Ridge, lambda = %g)"%(_lambda), MSE(terrain.df['y'],terrain.X@terrain.beta))
    """
