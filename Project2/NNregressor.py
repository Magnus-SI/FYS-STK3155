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
        """
        plots the NN fit as a function of epochs
        polydeg: polynomial degree to make up the predictors
        noise: relative sigma noise
        epoch_arr: array of different epochs to compare
        newplot: if True, create a new figure
        """
        deg = int(polydeg)
        self.changenoise(noisefraq=noise)           #change noise
        self.changepolydeg(polydeg = (deg, deg))    #change polnomial degree
        errs = np.zeros(len(epoch_arr))
        self.FFNN.doreset = True                    #reset current fit before each new fit
        self.FFNN.Xf = self.X.shape[1]              #features of X
        #dftrain, dftest = np.split(self.df, [int(0.8*self.N)])  #train test split
        for i,epoch in enumerate(epoch_arr):
            self.epochs = int(epoch)                #update with new epoch coutn
            errs[i] = self.kfolderr(method = self.FFNN.fit) #cost using k=2,3,4,5 of kfold error
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
        """
        Splits data into training and test data, and uses bootstrap to resample the
        training data before fitting and evaluating error on test data.

        K: amount of resamples
        param_arr: array containing parameter values
        param_name: name of parameter as string, for use in self.parachanger
        """
        split = int(0.8*self.N)             #80-20 train-test split
        MSEs = np.zeros(len(param_arr))
        biass = np.zeros(len(param_arr))
        variances = np.zeros(len(param_arr))
        dftrain, dftest = np.split(self.df, [split])    #performs the split
        testinds = dftest.index                         #indices of test data
        y = dftest['y'].values                          #y
        msebest = 1e10                      #temporary worst value, for saving the best value
        self.FFNN.doreset = True                        #reset weights and biases of NN before each fit
        if not self.compnoisy:                          #compare to exact data instead of noisy.
            y = dftest['y_exact'].values
        for j,paramval in enumerate(param_arr):
            self.parachanger(param_name, paramval)      #change parameter
            ypreds = np.zeros((len(dftest), K))
            for i in range(K):
                df = dftrain.sample(frac=1.0, replace=True)     #resample training data
                #df = dftest    #use instead to train on test data, to evaluate training error
                self.fit(self.FFNN.fit, df)                     #fit on training data
                ypreds[:,i] = self.FFNN.predict(self.X[testinds])[0]    #predict on test data
            MSEs[j] = np.mean(np.mean((y-ypreds.transpose())**2, axis=0))   #MSE
            biass[j] = np.mean((y-np.mean(ypreds,axis=1))**2)               #bias
            variances[j] = np.mean(np.var(ypreds, axis=1))                  #variance

        plt.figure()
        if param_name== "hlayer":       #hlayer is list, so must be treated separately
            param_arr = np.arange(len(param_arr[0]), len(param_arr)+len(param_arr[0]))
        plt.title(r"%i data points, $log_{10}(\hat{\sigma}) = -1$"%self.N) #noise title is hardcoded, could be changed
        plt.plot(param_arr, MSEs, label="MSE")
        plt.plot(param_arr, biass, label="bias")
        plt.plot(param_arr, variances, label="variance")

        #plt.plot(param_arr, biass+variances,'--', label="bias+var")
        plt.legend()
        if self.cost=="MSE":
            plt.yscale("log")
        plt.xlabel(param_name)
        plt.ylabel(self.cost)
        plt.savefig("Frankefigs/%sbiasvar.pdf"%param_name)
        plt.show()


    def degvnoiseerr(self, polydegs, noises):
        """
        Create 2d seaborn heatmap of the relation between polynomial degree and noise.
        Uses a normal train-test split to calculate the error.
        polydegs: polynomial degrees to use
        noises: noises to use
        """
        self.cost = "R2"
        TestErrors = np.zeros((len(polydegs), len(noises)))
        #TrainErrors = np.zeros((len(polydegs), len(noises)))
        showvals = True                     #show values in heatmap
        self.FFNN.doreset = True            #reset before each fit
        for i,deg in enumerate(polydegs):
            self.parachanger("deg", deg)

            for j,noise in enumerate(noises):
                self.changenoise(noisefraq=noise)
                print(i,j)      #shows progress

                #TestErrors[i,j] = self.kfolderr(ks = np.arange(2,6), method = FNN.FFNN.fit)
                TestErrors[i,j] = self.traintesterr()       #train vs test error
                #TrainErrors[i,j] = self.traintesterr(testerr = False)#self.fiterr(method = FNN.FFNN.fit)
        f, ax1 = plt.subplots(1,1, figsize = (12,12))
        h1=sns.heatmap(data=TestErrors,annot=showvals,cmap='viridis',ax=ax1,xticklabels=np.around(np.log10(noises), 1), yticklabels=polydegs, vmin = 0.95, vmax = 1.0, fmt = '.3g')
        print(np.max(TestErrors), np.argmax(TestErrors))
        ax1.set_xlabel(r'$log_{10}(\sigma)$')
        ax1.set_ylabel('Polynomial degree')
        ax1.set_title(r'%s epochs %s Test %s, #datapoints = %i'%(self.epochs,"FFNN",self.cost, int(self.N*self.frac)))
        plt.show()

    def optparafinder(self, Nloops, noise, epochs, eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr, inits):
        """
        Makes an opyimal parameter search for values of eta, polydeg, number of batches,
        hidden layers and the parameter a in ReLU. Does this by finding optimal values
        of one at a time, fixing the others, looping over an amount of times

        Nloops: times to loop over the search. Since the parameters are reset to their
        found optimal values in order, one would expect at least 2 loops necessary to
        find the optimal solution
        noise: noise to add to the data
        epochs: integer value of epochs to train for
        eta_arr: array containing values of eta.
        polydeg_arr: array containing integer values corresponding to order of polynomial coefficients
        nbatch_arr: array containing integer values corresponding to batches to use in SGD
        hlayer_arr: array or list, containing lists describing the hidden layers, e.g [[], [30], [60,30]]
        relu_arr: array containing values of the parameter a in relu
        inits: initial values for the above parameters.
        """

        arrs = [eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr] #array containing values in search space
        labels = ['eta', 'deg', 'nbatch', 'hlayer', 'relu']
        for s,v in zip(labels,inits):       #chane parameters to initial values
            self.parachanger(s, v)
        optvals = inits
        self.FFNN.doreset = True        #reset after each fit
        self.changenoise(noise)         #change noise
        self.epochs = epochs            #set epochs
        self.cost = "R2"                #currently only used with R2 score as cost
        optinds = np.zeros((Nloops, len(labels))).astype(int)
        opterrs = np.zeros(Nloops)
        bestr2 = 0
        for i in range(Nloops):                     #goes through Nloops loops
            for j,arr in enumerate(arrs):           #goes through the different parameters
                err_arr = np.zeros(len(arr))
                for k,val in enumerate(arr):        #goes through the values in a parameter
                    print(i,j,k)
                    self.parachanger(labels[j], val)                #update parameter
                    #err_arr[k] = self.traintesterr(testerr = True)
                    #evaluate kfold error with k=2,3,4,5with updated parameter:
                    err_arr[k] = self.kfolderr(ks = np.array([2,3,4,5]), method = self.FFNN.fit)

                optind = np.argmax(err_arr)     #index of optimal value of the parameter
                optinds[i,j] = optind           #store index
                self.parachanger(labels[j], arr[optind])    #change to optimal value
            opterrs[i] = np.max(err_arr)        #error for the foudn optimal values this loop
        optcomb = np.argmax(opterrs)            #highest R2 score from all the loops
        optfin = optinds[optcomb]               #parameter indices that caused best R2 score
        #print("Optimal values:")
        for i,val in enumerate(optfin):
            self.parachanger(labels[i],arrs[i][val])    #update parameter to best found value
            with open("opt%.2f.txt"%(np.log10(noise)), "a") as w:   #store best fit in text file, named with relevant noise
                w.write("\n")
                w.write(labels[i] + ": " + str(arrs[i][val]))
        return optinds, opterrs


    def parachanger(self, label, val):
        """
        Changes a parameter of the FFNN
        label: name of parameter as string
        val: value to change parameter to
        """
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
    """
    Explores a search space of optimal parameters using FrankeNN.optparafinder
    Nloops: number of loops
    noise: nosie to use
    epochs: epochs of training
    """
    #First initialize franke function and NN:
    FNN = FrankeNN()
    FNN.compnoisy = False   #compare to actual data
    FNN.gendat(400, noisefraq = noise, Function = FrankeFunction, deg = (5,5), randpoints = True)
    FNN.hlayers = [30,15]; FNN.activation = ReLU(0.01); FNN.outactivation = ReLU(1.00); FNN.epochs = 10; FNN.nbatches = 5; FNN.eta = 0.1
    FNN.initNN()
    FNN.cost = "R2"

    #Define parameter search space
    eta_arr = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4])#np.array([0.1, 0.15, 0.2])#np.array([0.05, 0.1, 0.15, 0.2])
    polydeg_arr = np.arange(1,8)#np.arange(1,15)
    nbatch_arr = np.array([1, 5, 10, 15, 20])#np.array([10, 15])
    hlayer_arr = [[], [30], [60], [30,15], [60,30,15], [32,16,8], [16,8,4]]
    relu_arr = np.array([0.005, 0.01, 0.015, 0.02, 0.03])
    inits = [0.15, 5, 10, [30,15], 0.01]        #initial values of the parameters in the same order
    optinds, opterrs = FNN.optparafinder(Nloops, noise, epochs, eta_arr, polydeg_arr, nbatch_arr, hlayer_arr, relu_arr, inits)
    #finopt = optinds[-1]
    return FNN,optinds, opterrs

def multinoiseoptpara(noises, Nloops, epochs):
    """
    explore search space of parameter, using multiple values for the noise, calling
    the optparaexplorer function.

    noises: noises to use
    Nloops: number of loops in optimal parameter search
    epochs: epochs of training.
    """
    dats = []
    for noise in noises:
        dats.append(optparaexplorer(Nloops, noise, epochs))
    return dats     #return all data (optimal values also saved to text files, with names of different noise)

def optparainstance():
    """
    The call of multinoiseoptpara we used to preduce the results shown in the table
    in the article.
    """
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

def optparaMSEs():
    """
    Using the optimal parameters found from the optimal parameter search (see table in report),
    this function uses the kfold error with k=2,3,4,5 to find the corresponding values of the MSE.
    """
    FN1 = FrankeNN()
    FN1.compnoisy = False
    FN1.gendat(400, noisefraq=1e-2, Function = FrankeFunction, deg =(4,4), randpoints = True)
    FN1.hlayers = [60,30,15]; FN1.activation = ReLU(0.005); FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500; FN1.nbatches = 15; FN1.eta = 0.2
    FN1.cost = "MSE"
    FN1.initNN()
    FN1.FFNN.doreset = True

    noises = np.logspace(-3,-0.5, 11)
    orders = np.array([6,7,7,7,6,4,4,5,3,2,1])
    etas = np.array([0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3, 0.1])
    arelus = np.array([0.01, 0.02, 0.03, 0.03, 0.015, 0.03, 0.005, 0.03, 0.005, 0.02, 0.015])
    batches = np.array([20, 20, 20, 20, 20, 15, 15, 15, 20, 5, 5])
    MSES = np.zeros(11)
    for i in range(11):
        print(i)
        FN1.changenoise(noises[i])
        FN1.parachanger('deg', orders[i])
        FN1.parachanger('eta', etas[i])
        FN1.parachanger('relu', arelus[i])
        FN1.parachanger('nbatch', batches[i])
        MSES[i] = FN1.kfolderr(ks = np.arange(2,6), method = FN1.FFNN.fit)
    return MSES

def convergencefind(noises, polydeg):
    """
    Uses FrankeNN.multiepoch for different values of noise, to compare the convegence
    time as the noise varies. Used to create plot shown in the report
    """
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

def biasvarplotter(N, noise, param_arr, param_name):
    """
    Creates bias variance plot for selected parameter. Plots used in report

    N: data points of the Franke Function
    noise: sigma noise as fraction of max amplitude
    param_arr: array of parameter values to test for
    param_name: string containing paramter name for use in FrankeNN.parachanger(name, value)
    """
    FN1 = FrankeNN()
    FN1.compnoisy = False       #compare to actual data when evaluating error
    FN1.gendat(N, noisefraq = noise, Function = FrankeFunction, deg = (4,4), randpoints = True)
    FN1.hlayers = [60,30,15]    #found from optimal parameter search
    FN1.activation = ReLU(0.005)
    FN1.outactivation = ReLU(1.00)
    FN1.epochs = 500
    FN1.nbatches = 15
    FN1.eta = 0.2
    FN1.initNN()
    K = 20      #resamples
    FN1.biasvar(K, param_arr, param_name)

def biasvarplots():
    "creates bias var plots in the report"
    hlayers = [[], [64], [64,32], [64,32,16], [64,32,16,8], [64,32,16,8,4],  [64,32,16,8,4,2]]
    epochs = np.logspace(1,3,11).astype(int)
    etas = np.logspace(-2,0.5,10)
    batches = np.array([1, 5, 10, 20, 40, 100])
    labels = ['hlayer', 'epochs', 'eta', 'nbatch']
    arrs = [hlayers, epochs, etas, batches]
    N = 100; noise = 1e-1
    for label,param_arr in zip(labels, arrs):
        print(label)
        biasvarplotter(N, noise, param_arr, label)

def degvnoisevhlayererr():
    """
    Creates 2d seaborn heatmap of noise vs polynomial order, for different hidden layer shapes
    """
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
