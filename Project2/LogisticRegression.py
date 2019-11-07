import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def sigmoid(t):
    return 1/(1 + np.exp(-t))


class Logistic:
    def __init__(self):
        """
        Initialize with X and Y data, so far this assumes binary classification
        X has the dimensions n times p, where n is the number of datapoints
        and p is the number of predictors. y is a n-dimensional vector where each
        y-value corresponds to the row in X with the same index.
        """
        self.hasfit = False

    def predict(self,x):
        """
        Returns the probability for y=1, given a matrix of X-values
        (can also be a vector)
        """
        if len(x.shape) == 1:
            _x = x.reshape(1,len(x))
        else:
            _x = x
        return sigmoid(self.beta[0]+np.sum(_x*self.beta[1:],axis=1))

    def p(self,indices):
        """
        returns n-size vecor p with probabilities for the 1 outcome
        """
        return sigmoid(np.sum(self.X[indices]*self.beta,axis = 1))

    def cost(self,beta):
        """
        cost function (not used so far)
        """
        t = sigmoid(beta[0]+np.sum(X*beta[1:],axis = 1))
        return np.sum(y*t - np.log(1+np.exp(t)))

    def update_beta(self,indices):
        """
        updates beta values with the derivative of the cost function
        """
        beta = self.beta
        p_vec = self.p(indices)
        beta += self.eta*self.X[indices].T@(self.y[indices]-p_vec).T

    def fit(self,X,y,N,eta,M=None):
        """
        Fits the beta-coefficient

        The parameters (in order) are:
        X - X data matrix
        y - y data
        N - number of epochs
        eta - learning rate
        M - minibatchsize
        """
        self.N, self.np = X.shape
        self.np += 1
        self.X = np.ones(shape=(self.N,self.np))
        self.X[:,1:] = X
        if len(y.shape) == 2:
            self.y = y[:,0]
        elif len(y.shape) == 1:
            self.y = y
        self.beta = np.random.normal(size = self.np)
        if M == None:
            M_size = self.N
        elif M>self.N:
            print("Given minibatch size was larger than the number of datapoints.\n\
            Setting minibatch size to the size of the dataset")
            M_size = self.N
        else:
            M_size = M

        n_M = self.N//M_size # Number of minibatches
        minibatc_indices = np.split(np.arange(self.N),np.arange(1,n_M)*M_size)
        self.eta = eta
        self.indices = np.arange(self.N)
        for i in range(N):
            for j in range(n_M):
                ind = np.random.randint(n_M)
                self.update_beta(minibatc_indices[ind])
            np.random.shuffle(self.indices)
            self.X = self.X[self.indices]
            self.y = self.y[self.indices]
        self.hasfit = True


if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    indices = np.array([0,7]) # Using mean radius and mean concave points to predict
    # (Seemed reasonable from plot)
    X_data = cancer.data[:,indices]
    y_data = cancer.target

    N_train = 5000
    model = Logistic()
    model.fit(X_data,y_data,N_train,0.01,128)

    print(f"Training accuracy = {np.sum(np.round(model(X_data)) == y_data)/len(y_data)}")
    if np.all(np.round(model(X_data))>(1-1e-8)):
        print(f"All predictions = 1, looks like something went wrong")
    elif np.all(np.round(model(X_data))==0):
        print(f"All predictions = 0, looks like something went wrong")


    n_train = 1000; p = 1
    X_train = np.random.random((n_train,p))
    y_train = np.array(X_train>0.5).astype(float)[:,0]

    n_test = 1432
    X_test = np.round(np.random.random((n_test,p)))
    y_test = np.array(X_test>0.5).astype(float)[:,0]

    X_test_noround = np.random.random((n_train,p))
    y_test_noround = np.array(X_test_noround>0.5).astype(float)[:,0]

    test_model = Logistic()
    test_model.fit(X_train,y_train,5000,0.01,128)

    train_acc = np.sum(np.round(test_model.predict(X_train)) == y_train)/len(y_train)
    test_acc = np.sum(np.round(test_model.predict(X_test)) == y_test)/len(y_test)
    test_acc_noround = np.sum(np.round(test_model.predict(X_test_noround)) == y_test_noround)/len(y_test_noround)

    print(f"Training accuracy test model                = {train_acc:.3f}")
    print(f"Testing accuracy test model (with rounding) = {test_acc:.3f}")
    print(f"Testing accuracy test model  (no rounding)  = {test_acc_noround:.3f}")

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
