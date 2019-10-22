import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn import datasets
from autograd import elementwise_grad as egrad

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
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims = True)

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
        self.eta = 1.0
        self.cost = cost # cost function
        self.dcda = egrad(cost,0) # partial derivtive of the cost function (with regards to a)
        self.dfdx = egrad(activation) # derivative of activation function
        self.dodx = egrad(outactivation)
        self.ah = [0] * (len(hlayers)+1) # list of a-vectors
        self.zh = [0] * (len(hlayers)+1) # list of z-vectors

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
        # X = df[X_vars]
        # y = df[y_vars]
        self.weights = [1e-2*np.random.uniform(0,1, size = (layers[i], layers[i+1]))
                        for i in range(len(layers)-1)]

        self.biass = [np.ones((layers[i]))*0.01
                        for i in range(1, len(layers))]

    def feedforward(self):
        clayer = self.dftrain[self.X_vars].values
        # print(clayer)
        # print(sigmoid(clayer))
        self.ah[0] = clayer

        for i in range(len(self.hlayers)):      #propagate through hidden layers
            zh = clayer@self.weights[i] + self.biass[i]
            self.zh[i+1] = zh
            # print(zh)
            clayer = self.activation(zh)
            self.ah[i+1] = clayer
            # print(clayer)

        z_out = clayer@self.weights[-1] + self.biass[-1]
        self.z_out = z_out
        self.out = self.outactivation(z_out)

    def backpropagate(self):
        eta, weights, biass = self.eta, self.weights, self.biass
        delta = [0]*(len(self.hlayers)+1)
        delta[-1] = self.outactivation(self.z_out)*(1 - self.outactivation(self.z_out))*self.dcda(self.out,self.dftrain[self.y_vars].values)
        for i in range(len(self.hlayers)):
            delta[-2-i] = (delta[-1-i]@weights[-1-i].T) * self.dfdx(self.zh[-1-i])
        for i in range(len(self.hlayers)):
            self.weights[-1-i] -= eta * self.ah[-1-i].T@delta[-1-i]
            self.biass[-1-i] -= eta * np.sum(delta[-1-i],axis = 0)

    def train(self, n_epochs):
        for n in range(n_epochs):
            self.feedforward()
            self.backpropagate()

    def testpredict(self):
        pass

    def error(self):
        pass

def gradientmethod():
    pass




if __name__ == "__main__":
    N1 = FFNN(hlayers = [50, 40, 20], activation = aRELU(0.1), outactivation = softmax, cost = cost_regression)
    N1.train(100)
    print(N1.out)
