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
        self.activation = ReLU(0.01)
        self.outactivation = ReLU(1.00)
        self.hlayers = [30,15]
        self.eta = 0.1
        self.epochs = 1000
        self.nbatches = 10
        #self.initNN()

    def initNN(self):
        Xfeatures = self.X.shape[1]
        yfeatures = 1
        self.FFNN = FFNN(self.hlayers, self.activation, self.outactivation, NN_MSE(), Xfeatures, yfeatures, self.eta)

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

    def traintesterr(self, trainfrac = 0.8, testerr = True):
        dftrain, dftest = np.split(self.df, [int(0.8*self.N)])
        self.fit(self.FFNN.fit, dftrain)
        if testerr:
            return self.testeval(dftest)
        else:
            return self.testeval(dftrain)

    def multitrain(self,N, epN):        #actually uses some strange sort of spread training
        self.FFNN.doreset = False
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

    def multiepoch(self, polydeg, noise, epoch_arr):
        deg = int(polydeg)
        self.changenoise(noisefraq=noise)
        self.changepolydeg(polydeg = (deg, deg))
        errs = np.zeros(len(epoch_arr))
        self.FFNN.doreset = True
        self.FFNN.Xf = self.X.shape[1]
        dftrain, dftest = np.split(self.df, [int(0.8*self.N)])
        for i,epoch in enumerate(epoch_arr):
            self.epochs = int(epoch)
            #errs[i] = self.kfolderr(method = self.FFNN.fit)
            errs[i] = self.traintesterr()
        plt.figure()
        plt.title(r"$log10(\sigma)$=%.1f, polynomial order = $%i$"%(np.log10(noise), deg))
        plt.plot(epoch_arr, errs)
        plt.xlabel("epoch")
        plt.ylabel("%s"%self.cost)
        if self.cost == "MSE":
            self.yscale("log")
        plt.show()


    def degvnoiseerr(self, polydegs, noises):
        TestErrors = np.zeros((len(polydegs), len(noises)))
        TrainErrors = np.zeros((len(polydegs), len(noises)))
        showvals = True
        self.FFNN.doreset = True
        self.epochs = 1000
        for i,deg in enumerate(polydegs):
            self.changepolydeg((deg,deg))
            self.initNN()

            for j,noise in enumerate(noises):
                self.changenoise(noisefraq=noise)
                print(i,j)      #shows progress

                #TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = FNN.FFNN.fit)
                TestErrors[i,j] = self.traintesterr()
                TrainErrors[i,j] = self.traintesterr(testerr = False)#self.fiterr(method = FNN.FFNN.fit)
        f, axs = plt.subplots(2,1, figsize=(12,12))
        ax1, ax2 = axs
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.95, vmax = 1.0, fmt = '.3g')
        ax1.set_xlabel(r'$log_{10}(\sigma)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title(r'%s epochs %s Test %s, #datapoints = %i'%(self.epochs,"FFNN",self.cost, int(self.N*self.frac)))
        h2=sns.heatmap(data=TrainErrors,annot=showvals,cmap='viridis',ax=ax2,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.95, vmax = 1.0, fmt = '.3g')
        ax2.set_xlabel(r'$log_{10}(\sigma)$')
        ax2.set_ylabel('Polynomial degree')
        ax2.set_title(r'%s epochs %s Train %s'%(self.epochs,"FFNN", self.cost))
        plt.show()

    def optparafinder(self, Nloops, noise, epochs, eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr, inits):
        arrs = [eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr]
        labels = ['eta', 'deg', 'nbatch', 'hlayer', 'relu']
        for s,v in zip(labels,inits):
            self.parachanger(s, v)
        optvals = inits
        self.FFNN.doreset = True
        self.changenoise(noise)
        self.epochs = epochs
        optinds = np.zeros((Nloops, len(labels))).astype(int)
        opterrs = np.zeros(Nloops)
        for i in range(Nloops):
            for j,arr in enumerate(arrs):
                err_arr = np.zeros(len(arr))
                for k,val in enumerate(arr):
                    print(i,j,k)
                    self.parachanger(labels[j], val)
                    #err_arr[k] = self.traintesterr(testerr = True)
                    err_arr[k] = self.kfolderr(ks = np.array([5]), method = self.FFNN.fit)
                optind = np.argmax(err_arr)
                optinds[i,j] = optind
                self.parachanger(labels[j], arr[optind])
            opterrs[i] = np.max(err_arr)
        optfin = optinds[-1]
        print("Optimal values:")
        for i,val in enumerate(optfin):
            self.parachanger(labels[i],arrs[i][val])
            print(labels[i] + ": " + str(arrs[i][val]))
        return optinds, opterrs


    def parachanger(self, label, val):
        if label == "eta":
            self.eta = val
            self.FFNN.eta = val
        elif label == "deg":
            self.changepolydeg((val, val))
            self.initNN()
        elif label == "nbatch":
            self.nbatches = val
        elif label == "hlayer":
            self.hlayers = val
            self.initNN()
        elif label == "relu":
            self.activation = ReLU(val)
            self.FFNN.activation = ReLU(val)

def optparaexplorer(Nloops, noise, epochs):
    FNN = FrankeNN()
    FNN.compnoisy = False
    FNN.gendat(400, noisefraq = noise, Function = FrankeFunction, deg = (5,5), randpoints = True)
    FNN.hlayers = [30,15]; FNN.activation = ReLU(0.01); FNN.outactivation = ReLU(1.00); FNN.epochs = 10; FNN.nbatches = 5; FNN.eta = 0.1
    FNN.initNN()
    FNN.cost = "R2"

    eta_arr = np.array([0.1, 0.15, 0.2])#np.array([0.05, 0.1, 0.15, 0.2])
    polydeg_arr = np.arange(1,15)
    nbatch_arr = np.array([10, 15])
    hlayer_arr = [[], [30], [30,15], [60,30,15]]
    relu_arr = np.array([0.005, 0.01, 0.015, 0.02])
    inits = [0.15, 14, 5, [30,15], 0.01]
    optinds, opterrs = FNN.optparafinder(Nloops, noise, epochs, eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr, inits)
    finopt = optinds[-1]
    return FNN,optinds, opterrs

if __name__ == "__main__":
    pass
    Nloops = 3
    noise = 3e-3
    epochs = 300
    #FNN, optinds, opterrs = optparaexplorer(Nloops, noise, epochs)
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(400, noisefraq=1e-2, Function = FrankeFunction, deg =(6,6), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00); FN1.epochs = 10; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.initNN()
    FN1.cost = "R2"
    degs = np.arange(1,6)
    noises = np.logspace(-2.5,-0.5,5)
    FN1.degvnoiseerr(degs, noises)

    #
    # polydegs = np.arange(1,15)
    # noises = np.logspace(-5,-1, 9)
    # FN1.cost = "R2"
    # #FN1.degvnoiseerr(polydegs, noises)
    # kf = FNN.kfolderr(method = FNN.FFNN.fit)
    #FNN.cost = "R2"
    #FNN.multitrain(50,1)
    #FNN.multiepoch(polydeg=6, noise=1e-2, epoch_arr = np.array([5,10,20,50]))
    #FNN.FFNN.eta = 0.2
    #FNN.degvnoiseerr(polydegs, noises)
