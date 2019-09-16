import numpy as np

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
        lmb_matrix = np.identity(X.shape[1])*lambda_
        return np.linalg.inv(np.transpose(X)@X+lmb_matrix)@np.transpose(X)@y

if __name__ == '__main__':
    pass