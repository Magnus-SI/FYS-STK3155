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
import matplotlib.pyplot as plt

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
        # self.FFNN.feedforward(self.X[inds])
        # y_pred = self.FFNN.out[0]
        y_pred = self.FFNN.predict(self.X[inds])[0]
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

    def multitrain(self,N, epN):
        errs = np.zeros(N)
        self.epochs = epN
        for i in range(N):
            errs[i] = self.kfolderr(method = self.FFNN.fit)
        print("Best R2: %.5f"%(np.max(errs)))
        plt.figure()
        plt.plot(np.arange(N)*epN, errs)
        plt.xlabel("epoch")
        plt.ylabel(self.cost)
        plt.show()

    def degvnoiseerr(self, polydegs, noises):
        TestErrors = np.zeros((len(polydegs), len(noises)))
        TrainErrors = np.zeros((len(polydegs), len(noises)))
        showvals = True
        for i,deg in enumerate(polydegs):
            self.changepolydeg((deg,deg))
            for j,noise in enumerate(noises):
                self.changenoise(noisefraq=noise)
                self.initNN([30,15], ReLU(0.01), ReLU(1.00), epochs=200, nbatches=10)
                print(i,j)      #shows progress
                TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = FNN.FFNN.fit)
                TrainErrors[i,j] = self.fiterr(method = FNN.FFNN.fit)
        f, axs = plt.subplots(2,1, figsize=(12,12))
        ax1, ax2 = axs
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.7, vmax = 1.0)
        ax1.set_xlabel(r'$log_{10}(\sigma)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title(r'%s Test %s, #datapoints = %i$'%("FFNN",self.cost, int(self.N*self.frac)))
        h2=sns.heatmap(data=TrainErrors,annot=showvals,cmap='viridis',ax=ax2,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.7, vmax = 1.0)
        ax2.set_xlabel(r'$log_{10}(\sigma)$')
        ax2.set_ylabel('Polynomial degree')
        ax2.set_title(r'%s Train %s'%("FFNN", self.cost))
        plt.show()

if __name__ == "__main__":
    FNN = FrankeNN()
    FNN.compnoisy = False
    FNN.gendat(1000, noisefraq=1e-2, Function = FrankeFunction, deg =(6,6), randpoints = True)
    FNN.initNN(hlayers = [30,15], activation = ReLU(0.01), outactivation = ReLU(1.00), epochs=10, nbatches = 5)

    polydegs = np.arange(1,15)
    noises = np.logspace(-5,-1, 9)
    FNN.cost = "R2"
    #FNN.degvnoiseerr(polydegs, noises)
    kf = FNN.kfolderr(method = FNN.FFNN.fit)
    #FNN.cost = "R2"
    FNN.multitrain(100,1)
