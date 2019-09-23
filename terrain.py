from project1 import idk, OLS
from imageio import imread
import pandas as pd
import numpy as np
from Ridge import Ridge

class Terrain(idk):
    def set_data(self,data,deg = (5,5)):
        """
        Set the data
        """
        self.polydeg = deg
        df = pd.DataFrame()
        self.N = data.shape[0]*data.shape[1]

        x1, x2 = np.meshgrid(np.arange(data.shape[0]),np.arange(data.shape[1]))
        x1 = x1.flatten()
        x2 = x2.flatten()
        y = data.flatten()

        df['x1'] = x1
        df['x2'] = x2
        df['y'] = y
        self.df = df
        self.changepolydeg(polydeg = deg)

if __name__ == '__main__':
    filename = "SRTM_data_Norway_1.tif"
    terrain_data = imread(filename)

    _lambda = 1e-2
    R = Ridge(_lambda)
    terrain = Terrain()
    terrain.set_data(terrain_data)
    terrain.fit(OLS)
    print("betas (OLS) = ",terrain.beta)

    terrain.fit(R)
    print("betas (Ridge, lambda = %g)"%(_lambda), terrain.beta)