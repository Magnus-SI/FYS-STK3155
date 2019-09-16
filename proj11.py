import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.polynomial.polynomial import polyvander2d

fig=plt.figure()
ax=fig.gca(projection='3d')


"Make data for plot:"
x1 = np.arange(0, 1, 0.05)
x2 = np.arange(0, 1, 0.05)
deg = 5 #Degree of the approximation
X = polyvander2d(x1,x2,deg)
x,y = np.meshgrid(x1,x2)

def FrankeFunction(x1,x2):
    term1 = 0.75*np.exp(-(0.25*(9*x1-2)**2)-0.25*((9*x2-2)**2))
    term2 = 0.75*np.exp(-((9*x1+1)**2)/49.0 - 0.1*(9*x2+1))
    term3 = 0.5*np.exp(-(9*x1-7)**2/4.0 - 0.25*((9*x2-3)**2))
    term4 = -0.2*np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    return term1 + term2 + term3 + term4

def OLS(X,y):
    return np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y

z=FrankeFunction(x,y)
"Plot surface:"
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.1, 1.4)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


"Make data for analysis:"

N=100       #Amount of data points
lr=LinearRegression(fit_intercept=False)
x,y=np.random.uniform(0,1,size=(2,N))      #randomly generated points on the xy-grid
z=FrankeFunction(x,y)
sigma=0.02*np.max(z)
mu=0
noise=np.random.normal(mu,sigma,size=N)
z+=noise
betas = OLS(X,z)