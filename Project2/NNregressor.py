import sys
sys.path.insert(1, '../Project1')
import numpy as np
from project1 import *
import NeuralNet as NN
import importlib
importlib.reload(NN)
from NeuralNet import FFNN
from Functions import MSE as NN_MSE
from Functions import ReLU

class FrankeNN(Project1):
    def __init__(self, seed = 2):
        np.random.seed(seed)
        self.data = False
        self.hasfit = False
        self.compnoisy = True
        self.cost = "MSE"
        self.frac = 1.0
        self.noexact = False

    def initNN(self, hlayers, activation, outactivation, epochs, nbatches):
        Xfeatures = self.X.shape[1]
        yfeatures = 1
        self.FFNN = FFNN(hlayers, activation, outactivation, NN_MSE(), Xfeatures, yfeatures)
        self.epochs = epochs
        self.nbatches = nbatches

    def fit(self, method, df):
        y = df['y'].values
        #print(y)
        yext = y[np.newaxis].T
        inds = df.index
        method(self.X[inds], yext, self.epochs, self.nbatches)
        self.hasfit = True

    def testeval(self, dftest):
        if not self.hasfit:
            print("Error : run fit before testeval")
            sys.exit(1)
        inds = dftest.index
        self.FFNN.feedforward(self.X[inds])
        y_pred = self.FFNN.out[0]
        if self.compnoisy:
            y = dftest['y']
        else:
            y = dftest['y_exact']
        N = len(y)
        if self.cost == "MSE":
            MSE = 1/N * np.sum((y_pred - y)**2)
            return MSE
        elif self.cost == "R2":
            score = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            return score
        else:
            print("Choose from MSE or R2 as a cost function")
            sys.exit(1)

if __name__ == "__main__":
    FNN = FrankeNN()
    FNN.gendat(500, noisefraq=0.05, Function = FrankeFunction, deg =(2,2), randpoints = True)
    FNN.initNN(hlayers = [], activation = ReLU(0.01), outactivation = ReLU(1.00), epochs=10, nbatches = 5)
    kf = FNN.kfolderr(method = FNN.FFNN.train)
