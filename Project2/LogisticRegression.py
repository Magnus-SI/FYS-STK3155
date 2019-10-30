from autograd import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def sigmoid(t):
    return 1/(1 + np.exp(-t))


class Logistic:
    def __init__(self,X,y):
        """
        Initialize with X and Y data, so far this assumes binary classification
        X has the dimensions n times p, where n is the number of datapoints
        and p is the number of predictors. y is a n-dimensional vector where each
        y-value corresponds to the row in X with the same index.
        """
        self.X, self.y = X,y
        self.N, self.np = X.shape
        self.beta = np.random.normal(size = self.np+1)
        self

    def __call__(self,x):
        """
        Returns the probability for y=1, given a matrix of X-values
        (can also be a vector)
        """
        if len(x.shape) == 1:
            _x = x.reshape(1,len(x))
        else:
            _x = x
        return sigmoid(self.beta[0]+np.sum(_x*self.beta[1:],axis=1))

    def p(self):
        """
        returns n-size vecor p with probabilities for the 1 outcome
        """
        beta, X = self.beta, self.X
        return sigmoid(beta[0]+np.sum(X*beta[1:],axis = 1))

    def cost(self,beta):
        """
        cost function (not used so far)
        """
        t = sigmoid(beta[0]+np.sum(X*beta[1:],axis = 1))
        return np.sum(y*t - np.log(1+np.exp(t)))

    def update_beta(self):
        """
        updates beta values with the derivative of the cost function
        """
        beta = self.beta
        p_vec = self.p()
        #beta[0] += self.eta*np.sum(self.y-p_vec)/self.N
        beta[1:] += self.eta*self.X.T@(self.y-p_vec).T/self.N

    def fit(self,N,eta):
        self.eta = eta
        for i in range(N):
            self.update_beta()

if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    indices = np.array([0,7]) #Using mean radius and mean concave points to predict
    # (Seemed reasonable from plot)
    X_data = cancer.data[:,indices]
    y_data = cancer.target

    N_train = 20000
    model = Logistic(X_data,y_data)
    model.fit(N_train,0.01)
    print(f"Training accuracy = {np.sum(np.round(model(X_data)) == y_data)/len(y_data)}")
    if np.all(np.round(model(X_data))>(1-1e-8)):
        print(f"All predictions = 1, looks like something went wrong")
    elif np.all(np.round(model(X_data))==0):
        print(f"All predictions = 0, looks like something went wrong")


    """
    correlation_matrix = cancerpd.corr().round(1)
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.show()

    fig, axes = plt.subplots(15,2,figsize=(10,20))
    malignant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]
    ax = axes.ravel()

    for i in range(30):
        _, bins = np.histogram(cancer.data[:,i], bins =50)
        ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
        ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["Malignant", "Benign"], loc ="best")
    fig.tight_layout()
    plt.show()
    """





