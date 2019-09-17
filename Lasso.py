import numpy as np
from sklearn.linear_model import Lasso as Lasso_
import matplotlib.pyplot as plt

class Lasso():
    def __init__(self,lambda_):
        self.lambda_ = lambda_
        self.clf = Lasso_(alpha = lambda_,fit_intercept = False)

    def __call__(self,X,y):
        """
        Fits the coefficient in beta the data passed

        Parameters:
        X : The design matrix
        y : The datapoints
        """
        self.clf.fit(X,y)
        return self.clf.coef_

if __name__ == "__main__":
    N = 500
    x_stop = 2

    x = x_stop*np.random.random(500)
    noise = 2.0
    y = 1 + 2*x + 10 * x**2 + noise*np.random.normal(size = N)
    plt.plot(x,y,'.')
    lambda_ = 0.2
    model = Lasso(lambda_)
    deg = 2
    X = np.polynomial.polynomial.polyvander(x,deg)
    betas = model(X,y)
    xi = np.linspace(0,x_stop,1000)
    print(betas)
    plt.plot(xi,betas[0] + xi * betas[1] + xi**2 * betas[2])
    plt.show()
