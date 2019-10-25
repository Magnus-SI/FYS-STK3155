import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn import datasets
from autograd import elementwise_grad as egrad
from autograd import jacobian, grad
import tensorflow as tf

def jacobian_diagonal(f,dim):
    """
    Returns a list (?) of the functions to get the derivatives on the
    diagonal of the jacobian matrix
    """
    f_i = lambda i: lambda *args: f(*args)[None,i]
    def derivative(*args):
        arr = np.array(*args)
        np.array([egrad(f_i(i))(arr[None,i]) for i in range(dim)])
    return derivative

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
    df = pd.DataFrame()
    df['x1'] = np.random.uniform(0,1, size=100)
    df['x2'] = np.random.uniform(0,1, size=100)
    df['y'] = df['x1']**2 - df['x2']
    X_vars = ['x1', 'x2']
    y_vars = ['y']
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

def softmax(x):
    """
    x: array of shape nxp where n contains different data points, and p
    describes different nodes in layer
    """
    return np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)

def cost_regression(value,target):
    """
    General cost function for regression
    value and target should be one dimensional arrays, with equal length
    """
    return 0.5*np.sum((value-target)**2)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class aRELU:
    def __init__(self, a):
        self.a = a
    def __call__(self, x):
        return x*(x>0) + self.a*x*(x<=0)


class FFNN:
    def __init__(self, hlayers, activation, outactivation, cost):
        """
        hlayers: list of hidden layer, e.g. [50, 20]
        """
        self.hlayers = np.array(hlayers).astype(int)
        self.dataload(testletter)
        self.traintest()
        self.NNinit()
        self.activation = activation    #function
        self.outactivation = outactivation
        self.eta = 0.1
        self.cost = cost # cost function
        self.dcda = egrad(cost,0) # partial derivtive of the cost function (with regards to a)
        self.dfdx = egrad(activation) # derivative of activation function
        if outactivation == softmax:
            self.dodx = jacobian_diagonal(outactivation,10) # TODO: Find way not hardcode
        else:
            self.dodx = egrad(outactivation)
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
        delta[-1] = self.out*(1-self.out)*(self.out-self.dftrain[self.y_vars].values.T)
        # Calculate delta values
        for i in range(self.N_layers-1):
            delta[-2-i] = (weights[-1-i].T@delta[-1-i]) * self.dfdx(self.ah[-1-i])
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

    def error(self):
        pass

def gradientmethod():
    pass




if __name__ == "__main__":
    N1 = FFNN(hlayers = [30,15], activation = aRELU(0.01), outactivation = softmax, cost = cost_regression)
    N1.train(5000)
    N1.feedforward()
    #print(N1.out)
    predscore = N1.testpredict()
    print("Fraction of correct guesses =", predscore)
    N_test = len(N1.dftest[N1.y_vars].values)
    print(f"{int(round(predscore*N_test)):d} correct out of {N_test} testing datapoints")
    print(f"Training accuracy = {N1.trainpredict()}")
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
