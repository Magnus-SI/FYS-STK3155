import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class FFNN:
    def __init__(self, hlayers, ativation):
        """
        hlayers: list of hidden layer, e.g. [50, 20]
        """
        self.hlayers = np.array(hlayers).astype(int)
        self.dataload()
        self.activation = activation    #function

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
        layers = [len(X_vars)] + list(self.hlayers) + len(y_vars)
        # X = df[X_vars]
        # y = df[y_vars]
        self.weights = [np.random.uniform(0,1, size = (layers[i], layers[i+1])
                        for i in range(len(layers)-1)]
        self.biass = [np.ones((size))*0.01
                        for size in self.hlayers]

    def feedforward(self):
        for i in range(len(self.hlayers)):
            pass

    def backpropagate(self):
        pass

    def train(self, n_epochs):
        pass

    def error(self):
        pass

def gradientmethod():
    pass




if __name__ == "__main__":
    def testdat():
        df = pd.DataFrame()
        df['x1'] = np.random.uniform(0,1, size=100)
        df['x2'] = np.random.uniform(0,1, size=100)
        df['y'] = df['x1']**2 - df['x2']
        X_vars = ['x1', 'x2']
        y_vars = ['y']
        return df, X_vars, y_vars
