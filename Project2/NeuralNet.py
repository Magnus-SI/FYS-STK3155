import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

def sigmoid(x):
    return 1/(1+np.exp(-x))

class aRELU:
    def __init__(self, a):
        self.a = a
    def __call__(self, x):
        return x*(x>0) + self.a*x*(x<=0)


class FFNN:
    def __init__(self, hlayers, activation, outactivation):
        """
        hlayers: list of hidden layer, e.g. [50, 20]
        """
        self.hlayers = np.array(hlayers).astype(int)
        self.dataload(testclassify)
        self.traintest()
        self.NNinit()
        self.activation = activation    #function
        self.outactivation = outactivation

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

    def NNinit(self):
        df, X_vars, y_vars = self.dftrain, self.X_vars, self.y_vars
        layers = [len(X_vars)] + list(self.hlayers) + [len(y_vars)]
        # X = df[X_vars]
        # y = df[y_vars]
        self.weights = [np.random.uniform(0,1, size = (layers[i], layers[i+1]))
                        for i in range(len(layers)-1)]
        self.biass = [np.ones((layers[i]))*0.01
                        for i in range(1, len(layers))]

    def feedforward(self, train = True):
        if train:
            clayer = self.dftrain[self.X_vars].values
        else:
            clayer = self.dftest[self.X_vars].values

        for i in range(len(self.hlayers)):      #propagate through hidden layers
            zh = clayer@self.weights[i] + self.biass[i]
            clayer = self.activation(zh)

        z_out = clayer@self.weights[-1] + self.biass[-1]
        self.out = self.outactivation(z_out)

    def backpropagate(self):
        pass

    def train(self, n_epochs):
        pass

    def error(self):
        pass

def gradientmethod():
    pass




if __name__ == "__main__":
    N1 = FFNN(hlayers = [50, 20], activation = aRELU(0.01), outactivation = softmax)
    N1.feedforward()
    print(self.out)
