import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.tri as mtri

class idk:
    def __init__(self, seed=1):
        np.random.seed(seed)
        self.data=False
        pass

    def FrankeFunction(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def gendat(self,N,noisefraq=0.05, Function=FrankeFunction):
        x1,x2 = np.random.uniform(0,1,size=(2,N))
        y_exact = self.FrankeFunction(x1,x2)
        sigma = np.max(y_exact)*noisefraq
        mu = 0
        y = y_exact + np.random.normal(sigma,mu,size=N)
        self.x1, self.x2, self.y_exact, self.y = x1, x2, y_exact, y
        self.N = N
        self.data=True

    def plot3D(self,approx=False):
        if not self.data:
            print("Generate data first")
            return
        triang=mtri.Triangulation(self.x1,self.x2)
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1, projection='3d')
        ax.plot_trisurf(triang,self.y,cmap='jet')
        ax.scatter(self.x1, self.x2, self.y, marker='.', s=10, c='black', alpha=0.5)
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
