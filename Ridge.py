import numpy as np
import matplotlib.pyplot as plt

class Ridge():
    def __init__(self,lambda_):
        """
        Parameters:
        lambda_ : Hyperparameter
        """
        self.lambda_ = lambda_

    def __call__(self,X,y):
        """
        Preforms Ridge regression, returns an array of estimates for the beta-vector

        Parameters:
        X : Design matrix
        y : Datapoints
        """
        lmb_matrix = np.identity(X.shape[1])*self.lambda_
        return np.linalg.inv(np.transpose(X)@X+lmb_matrix)@np.transpose(X)@y


if __name__ == "__main__":
    N = 500
    x_stop = 2

    x = x_stop*np.random.random(500)
    noise = 2.0
    act_beta = [1.0,2.0,10]
    y = act_beta[0] + act_beta[1]*x + act_beta[2] * x**2 + noise*np.random.normal(size = N)
    plt.plot(x,y,'.')
    lambda_ = 0.2
    model = Ridge(lambda_)
    deg = 2
    X = np.polynomial.polynomial.polyvander(x,deg)
    betas = model(X,y)
    xi = np.linspace(0,x_stop,1000)
    print("lambda = ", lambda_)
    print("Values for beta = ",betas,"\nActual values = ",act_beta)
    plt.plot(xi,betas[0] + xi * betas[1] + xi**2 * betas[2])
    plt.show()
