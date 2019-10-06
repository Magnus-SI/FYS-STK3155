import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

def inv_svd(X):
    """
    Caclculates the invers of a matrix A using numpys svd function
    """
    U, s, VT = np.linalg.svd(X)
    invD = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        invD[i,i]=1/s[i]
    UT = np.transpose(U); V = np.transpose(VT)
    return np.matmul(V,np.matmul(invD,UT))

class Ridge():
    def __init__(self,_lambda):
        """
        Parameters:
        _lambda : Hyperparameter
        """
        self._lambda = _lambda
        self.__name__ = "Ridge"

    def __call__(self,X,y):
        """
        Preforms Ridge regression, returns an array of estimates for the beta-vector

        Parameters:
        X : Design matrix
        y : Datapoints
        """
        lmb_matrix = np.identity(X.shape[1])*self._lambda
        return inv_svd(np.transpose(X)@X+lmb_matrix)@np.transpose(X)@y

class Ridgeskl(Ridge):
    def __call__(self,X,y):
        lr = skl.Ridge(self._lambda, fit_intercept=False)
        lr.fit(X,y)
        return lr.coef_


if __name__ == "__main__":
    N = 500
    x_stop = 2

    x = x_stop*np.random.random(500)
    noise = 2.0
    act_beta = [1.0,2.0,10]
    y = act_beta[0] + act_beta[1]*x + act_beta[2] * x**2 + noise*np.random.normal(size = N)
    plt.plot(x,y,'.')
    _lambda = 0.2
    model = Ridge(_lambda)
    deg = 2
    X = np.polynomial.polynomial.polyvander(x,deg)
    betas = model(X,y)
    xi = np.linspace(0,x_stop,1000)
    print("lambda = ", _lambda)
    print("Values for beta = ",betas,"\nActual values = ",act_beta)
    plt.plot(xi,betas[0] + xi * betas[1] + xi**2 * betas[2])
    plt.show()
