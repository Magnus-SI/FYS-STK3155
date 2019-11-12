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

class FrankeNN(Project1):       #inherits some functinos from the Project1 class
    def __init__(self, seed = 2):
        """
        Sets various parameters describing the data analysis, and the FFNN model within
        this class. These parameters can later be changed.
        seed: if specific random seed is desired
        """
        np.random.seed(seed)
        self.data = False       #data has not yet been generated
        self.hasfit = False     #a fit has not yet been made
        self.compnoisy = True   #compare to noisy/not noisy data when evaluating error
        self.cost = "MSE"       #cost function to use
        self.frac = 1.0         #fraction of data to use
        self.noexact = False    #True if no exact solution
        self.activation = ReLU(0.01)        #activation function on hidden layers of FFNN
        self.outactivation = ReLU(1.00)     #activation function on output node of FFNN
        self.hlayers = [30,15]              #hidden layers
        self.eta = 0.1                      #learning rate
        self.epochs = 1000                  #epochs
        self.nbatches = 10                  #batches for SGD
        #self.initNN()

    def initNN(self):
        """
        Initializes the NN based on the current count of features in self.X, along
        with the parameters initialized with the class.
        self.FFNN is the class containing the neural network.
        """
        Xfeatures = self.X.shape[1]
        yfeatures = 1
        self.FFNN = FFNN(self.hlayers, self.activation, self.outactivation, NN_MSE(), Xfeatures, yfeatures, self.eta)
        self.FFNN.doreset = True            #reset after each fit

    def fit(self, method, df):
        """
        fits the data in df using a given method.
        method: method to fit, in this case, this will be self.FFNN.fit. It is made generally to
        acommodate for the present functions inherited from the Project1 class.
        df: the dataframe containing the y-values to compare to.
        """
        y = df['y'].values
        #print(y)
        yext = y[np.newaxis].T      #y must be (n,1) for the NN to work
        inds = df.index
        method(self.X[inds], yext, self.epochs, self.nbatches)
        self.hasfit = True

    def testeval(self, dftest):
        """
        Evaluates the cost on the given dataframe dftest.
        dftest: dataframe containing the test data
        """
        if not self.hasfit:
            print("Error : run fit before testeval")
            sys.exit(1)
        inds = dftest.index
        # self.FFNN.feedforward(self.X[inds])
        # y_pred = self.FFNN.out[0]
        y_pred = self.FFNN.predict(self.X[inds])[0]
        if self.compnoisy:      #compare to noisy data
            y = dftest['y']
        else:
            y = dftest['y_exact']       #compare to exact data
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
        """
        performs a train test split and calculates the error
        trainfrac:  fraction of data to use for training
        testerr: if True, evaluate on test data, if False evaluate error on train data
        """


        dftrain, dftest = np.split(self.df, [int(0.8*self.N)])
        self.fit(self.FFNN.fit, dftrain)
        if testerr:
            return self.testeval(dftest)
        else:
            return self.testeval(dftrain)

    def multitrain(self,N, epN):        #actually uses some strange sort of spread training
        """
        Performs N loops of epN epochs of training per loop, evaluating the progression
        of the error as for more epochs. The fit is not reset, so due to the k-fold error,
        this issues some form of spread training, which is not exactly suitable to calculate
        train test error, but allows for the fastest convergence, to measure optimal fit.
        """
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

    def multiepoch(self, polydeg, noise, epoch_arr, newplot = True):
        deg = int(polydeg)
        self.changenoise(noisefraq=noise)
        self.changepolydeg(polydeg = (deg, deg))
        errs = np.zeros(len(epoch_arr))
        self.FFNN.doreset = True
        self.FFNN.Xf = self.X.shape[1]
        dftrain, dftest = np.split(self.df, [int(0.8*self.N)])
        for i,epoch in enumerate(epoch_arr):
            self.epochs = int(epoch)
            errs[i] = self.kfolderr(method = self.FFNN.fit)
            #errs[i] = self.traintesterr(testerr = False)
        if newplot:
            plt.figure()
        plt.title(r"400 data points, $\eta = %.2f$"%(self.eta))
        plt.plot(epoch_arr, errs, label = r"$log_{10}(\hat{\sigma})$=%.1f"%np.log10(noise))
        plt.xlabel("epoch")
        plt.ylabel("%s"%self.cost)
        if self.cost == "MSE":
            plt.yscale("log")
        plt.legend()
        plt.show()

    def biasvar(self, K, param_arr, param_name):
        split = int(0.2*self.N)             #80-20 train-test split
        MSEs = np.zeros(len(param_arr))
        biass = np.zeros(len(param_arr))
        variances = np.zeros(len(param_arr))
        dftrain, dftest = np.split(self.df, [split])    #performs the split
        testinds = dftest.index
        y = dftest['y'].values
        msebest = 1e10                      #temporary worst value, for saving the best value
        self.FFNN.doreset = True
        if not self.compnoisy:
            y = dftest['y_exact'].values
        for j,paramval in enumerate(param_arr):
            self.parachanger(param_name, paramval)
            ypreds = np.zeros((len(dftest), K))
            for i in range(K):
                #df = dftrain.sample(frac=1.0, replace=True)
                df = dftest
                self.fit(self.FFNN.fit, df)
                ypreds[:,i] = self.FFNN.predict(self.X[testinds])[0]
            MSEs[j] = np.mean(np.mean((y-ypreds.transpose())**2, axis=0))
            biass[j] = np.mean((y-np.mean(ypreds,axis=1))**2)
            variances[j] = np.mean(np.var(ypreds, axis=1))

        plt.figure()
        if param_name== "hlayer":
            param_arr = np.arange(1, len(param_arr)+1)
        plt.plot(param_arr, MSEs, label="MSE")
        plt.plot(param_arr, biass, label="bias")
        plt.plot(param_arr, variances, label="variance")
        plt.plot(param_arr, biass+variances,'--', label="bias+var")
        plt.legend()
        if self.cost=="MSE":
            plt.yscale("log")
        plt.xlabel(param_name)
        plt.ylabel(self.cost)
        plt.show()


    def degvnoiseerr(self, polydegs, noises):
        TestErrors = np.zeros((len(polydegs), len(noises)))
        TrainErrors = np.zeros((len(polydegs), len(noises)))
        showvals = True
        self.FFNN.doreset = True
        for i,deg in enumerate(polydegs):
            self.parachanger("deg", deg)

            for j,noise in enumerate(noises):
                self.changenoise(noisefraq=noise)
                print(i,j)      #shows progress

                #TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = FNN.FFNN.fit)
                TestErrors[i,j] = self.traintesterr()
                #TrainErrors[i,j] = self.traintesterr(testerr = False)#self.fiterr(method = FNN.FFNN.fit)
        #f, axs = plt.subplots(2,1, figsize=(12,12))
        #ax1, ax2 = axs
        f, ax1 = plt.subplots(1,1, figsize = (12,12))
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.95, vmax = 1.0, fmt = '.3g')
        print(np.max(TestErrors), np.argmax(TestErrors))
        ax1.set_xlabel(r'$log_{10}(\sigma)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title(r'%s epochs %s Test %s, #datapoints = %i'%(self.epochs,"FFNN",self.cost, int(self.N*self.frac)))
        # h2=sns.heatmap(data=TrainErrors,annot=showvals,cmap='viridis',ax=ax2,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.95, vmax = 1.0, fmt = '.3g')
        # ax2.set_xlabel(r'$log_{10}(\sigma)$')
        # ax2.set_ylabel('Polynomial degree')
        # ax2.set_title(r'%s epochs %s Train %s'%(self.epochs,"FFNN", self.cost))
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
        bestr2 = 0
        for i in range(Nloops):
            for j,arr in enumerate(arrs):
                err_arr = np.zeros(len(arr))
                for k,val in enumerate(arr):
                    print(i,j,k)
                    self.parachanger(labels[j], val)
                    #err_arr[k] = self.traintesterr(testerr = True)
                    err_arr[k] = self.kfolderr(ks = np.array([2,3,4,5]), method = self.FFNN.fit)

                optind = np.argmax(err_arr)
                optinds[i,j] = optind
                self.parachanger(labels[j], arr[optind])
            opterrs[i] = np.max(err_arr)
        optcomb = np.argmax(opterrs)
        optfin = optinds[optcomb]
        #print("Optimal values:")
        for i,val in enumerate(optfin):
            self.parachanger(labels[i],arrs[i][val])
            with open("opt%.2f.txt"%(np.log10(noise)), "a") as w:
                w.write("\n")
                w.write(labels[i] + ": " + str(arrs[i][val]))
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
        elif label == "epochs":
            self.epochs = val

def optparaexplorer(Nloops, noise, epochs):
    FNN = FrankeNN()
    FNN.compnoisy = False
    FNN.gendat(400, noisefraq = noise, Function = FrankeFunction, deg = (5,5), randpoints = True)
    FNN.hlayers = [30,15]; FNN.activation = ReLU(0.01); FNN.outactivation = ReLU(1.00); FNN.epochs = 10; FNN.nbatches = 5; FNN.eta = 0.1
    FNN.initNN()
    FNN.cost = "R2"

    eta_arr = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4])#np.array([0.1, 0.15, 0.2])#np.array([0.05, 0.1, 0.15, 0.2])
    polydeg_arr = np.arange(1,8)#np.arange(1,15)
    nbatch_arr = np.array([1, 5, 10, 15, 20])#np.array([10, 15])
    hlayer_arr = [[], [30], [60], [30,15], [60,30,15], [32,16,8], [16,8,4]]
    relu_arr = np.array([0.005, 0.01, 0.015, 0.02, 0.03])
    inits = [0.15, 5, 10, [30,15], 0.01]
    optinds, opterrs = FNN.optparafinder(Nloops, noise, epochs, eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr, inits)
    finopt = optinds[-1]
    return FNN,optinds, opterrs

def multinoiseoptpara(noises, Nloops, epochs):
    dats = []
    for noise in noises:
        dats.append(optparaexplorer(Nloops, noise, epochs))
    return dats

def optparainstance():
    Nloops = 3
    noise = 1e-2
    epochs = 500
    noises = np.logspace(-3,-0.5, 11)
    etas = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
    from time import time
    start = time()
    dat = multinoiseoptpara(noises, Nloops, epochs)
    end = time()
    return dat, (end-start)

def convergencefind(noises, polydeg):
    FNN = FrankeNN()
    FNN.compnoisy = False
    FNN.gendat(400, noisefraq = 1e-2, Function = FrankeFunction, deg = (4,4), randpoints = True)
    FNN.hlayers = [60,30,15]; FNN.activation = ReLU(0.01); FNN.outactivation = ReLU(1.00)
    FNN.epochs = 10; FNN.nbatches = 15; FNN.eta = 0.3
    FNN.initNN()
    FNN.cost = "MSE"
    #epoch_arr = np.array([10, 30, 100, 300])
    epoch_arr = np.logspace(1,4, 7)
    plt.figure()
    for noise in noises:
        print("yo")
        FNN.multiepoch(polydeg, noise, epoch_arr, newplot = False)
    plt.xscale("log")
    plt.legend()

def epochbiasvar():
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(100, noisefraq=1e-1, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.initNN()
    paramname = "epochs"; param_arr = np.logspace(1,3,11).astype(int)#np.array([10,30,100,300,1000])
    #paramname = "hlayer"; param_arr = [[64], [64, 32], [64,32,16], [64,32,16,8], [64,32,16,8,4]]
    K = 20

    FN1.biasvar(K, param_arr, paramname)

def etabiasvar():
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(100, noisefraq=1e-1, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.initNN()
    paramname = "eta"; param_arr = np.logspace(-2,0.5, 10)
    K = 20

    FN1.biasvar(K, param_arr, paramname)

def hlayerbiasvar():
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(400, noisefraq=1e-2, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.initNN()
    #paramname = "epochs"; param_arr = np.array([10,100,1000])
    paramname = "hlayer"; param_arr = [[64], [64, 32], [64,32,16], [64,32,16,8], [64,32,16,8,4]]
    K = 15
    FN1.biasvar(K, param_arr, paramname)

def batchbiasvar():
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(100, noisefraq=1e-1, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.initNN()
    #paramname = "epochs"; param_arr = np.array([10,100,1000])
    paramname = "nbatch"; param_arr = np.array([1, 5, 10, 20, 40, 100])
    K = 15
    FN1.biasvar(K, param_arr, paramname)

def degvnoisevhlayererr():
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(400, noisefraq=1e-1, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.cost = "R2"
    FN1.initNN()
    degs = np.arange(1,9)
    for hlayers in [[], [30], [30,15], [60,30,15]]:
        FN1.parachanger('hlayer', hlayers)
        FN1.degvnoiseerr(degs, np.logspace(-3, -0.5, 11))

if __name__ == "__main__":
    pass
    #dat, t_elapsed = optparainstance()  #WARNING: will take 5 hours-1 day depending on hardware.

    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(400, noisefraq=1e-2, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.cost = "R2"
    FN1.initNN()

    #convergencefind(np.logspace(-2, -0.5, 4), 4)
    #FN1.biasvar(10, np.arange(1,12))
    # FN1.cost = "R2"
    # #degs = np.arange(1,6)
    # #noises = np.logspace(-2.5,-0.5,5)
    # degs = np.array([1,2])
    # noises = np.array([1e-1])
    # FN1.degvnoiseerr(degs, noises)

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
