import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn import datasets
from autograd import elementwise_grad as egrad
from autograd import jacobian, grad
import tensorflow as tf
from Functions import MSE, ReLU, Softmax, Sigmoid
from numpy.polynomial.polynomial import polyvander2d
from sklearn.neural_network import MLPRegressor

def testletter():
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

def testlinreg():
    def FrankeFunction(x,y):
        "The Franke function"
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    df = pd.DataFrame()
    x1,x2 = np.random.uniform(0,1,size=(2,100))
    y = FrankeFunction(x1,x2)
    #X = polyvander2d(x1,x2, (10,10))
    # X_vars = ["%i"%i for i in range(X.shape[1])]
    # for i,label in enumerate(X_vars):
    #     df[label] = X[:,i]
    df['x1'] = x1
    df['x2'] = x2
    X_vars = ['x1', 'x2']
    y_vars = ['y']
    df['y'] = y
    return df, X_vars, y_vars

def testclassify():
    df = pd.DataFrame()
    N = 10
    minint = -3; maxint = 5
    x = np.random.randint(minint,maxint, size=N)
    onehot = np.zeros((N,maxint-minint))
    onehot[np.arange(N), x] = 1
    df['a'] = x**3
    df['b'] = x**2
    df['c'] = x**5
    X_vars = ['a', 'b', 'c']
    y_vars = [str(i) for i in range(minint, maxint)]
    for i,y in enumerate(y_vars):
        df[y] = onehot[:,i]
    return df, X_vars, y_vars


class FFNN:
    def __init__(self, hlayers, activation, outactivation, cost, loader):
        """
        hlayers: list of hidden layer, e.g. [50, 20]
        """
        self.hlayers = np.array(hlayers).astype(int)
        self.dataload(loader)
        self.traintest()
        self.NNinit()
        self.activation = activation    #function
        self.outactivation = outactivation
        self.eta = 0.1
        self.cost = cost # cost function
        self.ah = [0] * (len(hlayers)+1) # list of a-vectors
        self.zh = [0] * (len(hlayers)+1) # list of z-vectors
        self.delta = [0] * (len(hlayers)+1) # list of delta vectors


    def dataload(self, loader):
        """
        X_vars and y_vars can be used to extract input and output
        from a dataframe, allowing for simpler train-test splits
        """
        self.df, self.X_vars, self.y_vars = loader()
        self.N = len(self.df)

    def traintest(self, frac=0.8):
        df = self.df
        split = int(frac*self.N)             #80-20 train-test split
        self.dftrain, self.dftest = np.split(self.df, [split])    #performs the split
        self.N_train = self.dftrain[self.X_vars].values.shape[0]

    def NNinit(self):
        df, X_vars, y_vars = self.dftrain, self.X_vars, self.y_vars
        layers = [len(X_vars)] + list(self.hlayers) + [len(y_vars)]

        self.N_layers = len(layers)-1 # Number of layers with weights and biases

        # X = df[X_vars]
        # y = df[y_vars]
        scale_constant = 0.5e-0
        self.weights = [scale_constant*np.random.uniform(0,1, size = (layers[i+1], layers[i]))-scale_constant/2
                        for i in range(self.N_layers)]

        self.biass = [np.ones((layers[i])).T*0.01
                        for i in range(1, self.N_layers+1)]


    def feedforward(self, test =False):
        if test:
            clayer = self.dftest[self.X_vars].values.T# TEMP:
        else:
            clayer = self.dftrain[self.X_vars].values.T

        self.ah[0] = clayer
        for i in range(self.N_layers-1):      #propagate through hidden layers
            zh = self.weights[i]@clayer + self.biass[i][:,None]
            self.zh[i+1] = zh

            clayer = self.activation(zh)
            self.ah[i+1] = clayer

        z_out = self.weights[-1]@clayer + self.biass[-1][:,None]
        self.z_out = z_out
        self.out = self.outactivation(z_out)

    def backpropagate(self):
        eta, weights, biass, delta = self.eta, self.weights, self.biass, self.delta
        self.outactivation.derivative(self.z_out)
        delta[-1] = self.outactivation.derivative(self.z_out)*self.cost.derivative(self.out,self.dftrain[self.y_vars].values.T)
        # Calculate delta values
        for i in range(self.N_layers-1):
            delta[-2-i] = (weights[-1-i].T@delta[-1-i]) * self.activation.derivative(self.ah[-1-i])
        # Update weights and biases
        for i in range(self.N_layers):
            self.weights[-1-i] -= eta * delta[-1-i]@self.ah[-1-i].T/self.N_train
            self.biass[-1-i] -= eta * np.sum(delta[-1-i],axis = 1)/self.N_train

    def train(self, n_epochs):
        for n in range(n_epochs):
            self.feedforward()
            self.backpropagate()

    def testpredict(self):
        self.feedforward(test=True)
        prednums = np.argmax(self.out, axis = 0)
        nums = np.argmax(self.dftest[self.y_vars].values, axis = 1)
        return np.sum((prednums-nums)==0)/ len(nums)

    def trainpredict(self):
        self.feedforward()
        prednums = np.argmax(self.out, axis = 0)
        nums = np.argmax(self.dftrain[self.y_vars].values, axis = 1)
        return np.sum((prednums-nums)==0)/ len(nums)

    def testpredreg(self):
        self.feedforward(test=True)
        return 1/self.out.shape[1]*np.sum((self.out-self.dftest[self.y_vars].values.T)**2)

    def trainpredreg(self):
        self.feedforward()
        return 1/self.out.shape[1]*np.sum((self.out-self.dftrain[self.y_vars].values.T)**2)

    def error(self):
        pass

def gradientmethod():
    pass




if __name__ == "__main__":
    N1 = FFNN(hlayers = [100,50], activation = ReLU(0.01), outactivation = ReLU(1.00), cost = MSE(), loader = testlinreg)
    N1.train(1000)
    N1.feedforward()
    #print(N1.out)
    print(N1.testpredreg(), N1.trainpredreg())

    reg = MLPRegressor(hidden_layer_sizes = (100,50),
                        solver = 'lbfgs',
                        max_iter = 1000,
                        tol = 1e-7,
                        verbose = False)
    reg.fit(N1.dftrain[N1.X_vars].values, N1.dftrain[N1.y_vars].values[:,0])
    pred = reg.predict(N1.dftest[N1.X_vars].values)
    print(1/len(pred) * np.sum((pred - N1.dftest[N1.y_vars].values[:,0])**2))
    tpred = reg.predict(N1.dftrain[N1.X_vars].values)
    print(1/len(tpred) * np.sum((tpred - N1.dftrain[N1.y_vars].values[:,0])**2))


    "Regression tests below:"
    # predscore = N1.testpredict()
    # print("Fraction of correct guesses =", predscore)
    # N_test = len(N1.dftest[N1.y_vars].values)
    # print(f"{int(round(predscore*N_test)):d} correct out of {N_test} testing datapoints")
    # print(f"Training accuracy = {N1.trainpredict()}")
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
