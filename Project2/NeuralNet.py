import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn import datasets
import tensorflow as tf
from Functions import MSE, ReLU, Softmax, Sigmoid
from numpy.polynomial.polynomial import polyvander2d
from sklearn.neural_network import MLPRegressor

def testletter():
    """
    test case based on the MNIST letter data base
    """
    digits = datasets.load_digits()
    inputs = digits.images
    labels = digits.target
    N = inputs.shape[0]
    onehot = np.zeros((N,10))
    onehot[np.arange(N), labels] = 1
    X = inputs.reshape(N, inputs.shape[1]*inputs.shape[2])
    df = pd.DataFrame()
    X_vars = [str(i) for i in range(1,65)]
    y_vars = [str(i) for i in range(10)]
    for i,x in enumerate(X_vars):
        df[x] = X[:,i]
    for i,y in enumerate(y_vars):
        df[y] = onehot[:,i]

    return df, X_vars, y_vars

class FFNN:
    def __init__(self, hlayers, activation, outactivation, cost, Xfeatures, yfeatures,eta = 0.1):
        """
        Defines the parameters of the feed forward neural Network

        *Input*
        hlayers: list of hidden layer, e.g. [50, 20]
        activation: the activation function used on the nodes in the hidden layers
        outactivation: the activation function used in the output layer
        cost: the cost function, defined as a class with a derivative subfunction
        Xfeatures: the amount of predictors used
        yfeaturtes: the dimension of the output aimed to model, for regression this would be 1,
                    for binary classification, this would be 2.
        eta: the learning rate
        """
        self.hlayers = np.array(hlayers).astype(int)
        self.NNinit(Xfeatures, yfeatures)
        self.activation = activation    #function
        self.outactivation = outactivation
        self.cost = cost # cost function
        self.ah = [0] * (len(hlayers)+1) # list of a-vectors
        self.zh = [0] * (len(hlayers)+1) # list of z-vectors
        self.delta = [0] * (len(hlayers)+1) # list of delta vectors
        self.doreset = True                 #True if reset before each new fit
        self.eta = eta

    def NNinit(self, Xfeatures, yfeatures):
        """
        Initializes the FFNN, with the predictor count Xfeatures, and ouput
        dimension yfeatures as arguments.
        """
        self.Xf = int(Xfeatures); self.yf = int(yfeatures)
        layers = [self.Xf] + list(self.hlayers) + [self.yf]

        self.N_layers = len(layers)-1 # Number of layers with weights and biases

        # X = df[X_vars]
        # y = df[y_vars]
        scale_constant = 0.5e-0
        self.weights = [scale_constant*np.random.uniform(0,1, size = (layers[i+1], layers[i]))-scale_constant/2
                        for i in range(self.N_layers)]

        self.biass = [np.ones((layers[i])).T*0.01
                        for i in range(1, self.N_layers+1)]
        self.hasfit = True


    def feedforward(self, X):
        """
        feeds forward input X to output set as self.out
        X: the data to use as input. has shape [n,Xfeatures] with n data points
        self.out has shape (yfeatures, n)
        """
        clayer = X.T

        self.ah[0] = clayer
        for i in range(self.N_layers-1):      #propagate through hidden layers
            zh = self.weights[i]@clayer + self.biass[i][:,None]
            self.zh[i+1] = zh

            clayer = self.activation(zh)
            self.ah[i+1] = clayer

        z_out = self.weights[-1]@clayer + self.biass[-1][:,None]
        self.z_out = z_out
        self.out = self.outactivation(z_out)

    def backpropagate(self, y):
        """
        Backpropagetes through the network to update the weights and biases
        y: the true output compared to self.out gotten from self.feedforward,
            has shape [n,yfeatures] with n data points.
        """
        eta, weights, biass, delta = self.eta, self.weights, self.biass, self.delta
        self.outactivation.derivative(self.z_out)
        delta[-1] = self.outactivation.derivative(self.z_out)*self.cost.derivative(self.out,y.T)
        # Calculate delta values
        for i in range(self.N_layers-1):
            delta[-2-i] = (weights[-1-i].T@delta[-1-i]) * self.activation.derivative(self.ah[-1-i])
        # Update weights and biases
        for i in range(self.N_layers):
            self.weights[-1-i] -= eta * delta[-1-i]@self.ah[-1-i].T/y.shape[0]
            self.biass[-1-i] -= eta * np.sum(delta[-1-i],axis = 1)/y.shape[0]

    def fit(self, X, y, n_epochs, batches = 1, eta = None):
        """
        performs a fit by repeating feedforward and backpropagate for a given
        number of epochs, updating the biases and weights for each epoch.

        X: input data to use for feedforward, usually training data
        y: output data to compare result to, usually training data
        n_epochs: number of epochs to fit
        batches: number of batches to split, batches>1 -> SGD instead of GD
        eta: the learning rate eta may be reset, but usually done through self.eta
        """
        if self.doreset:
            self.reset()      #reset any previous fits
        if eta is not None:
            self.eta = eta    #reset eta
        allinds = np.arange(X.shape[0])
        batchinds = np.array_split(allinds, batches)

        for n in range(n_epochs):
            for j in range(batches):
                inds = batchinds[np.random.choice(range(batches))]
                self.feedforward(X[inds])
                self.backpropagate(y[inds])

    def predict(self, X):
        """
        Predicts on a given X-data based on the current weights and biases.
        X: predict on this data, usually test data.
        """
        self.feedforward(X)
        return self.out

    def reset(self):
        """
        Resets the weights and biases of the NN
        """
        self.NNinit(self.Xf, self.yf)



if __name__ == "__main__":
    pass
    "Regression tests below with scikit learn:"
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(40, activation = tf.keras.activations.relu),
            #tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation = tf.keras.activations.relu),
            tf.keras.layers.Dense(10, activation = 'softmax')
        ])
    # Compile model
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
        metrics = ['accuracy']
    )
    # Fit to training data
    print(N1.dftrain[N1.X_vars].values.T.shape)
    print(N1.dftrain[N1.y_vars].values.T.shape)
    model.fit(
        N1.dftrain[N1.X_vars].values,
        N1.dftrain[N1.y_vars].values,
        epochs = 1000,
        batch_size = 1437
    )
    print(model.predict(N1.dftrain[N1.X_vars].values[0,None]))
    """
