import numpy as np
from Analyze import ModelAnalysis
from NeuralNet import FFNN
from LogisticRegression import Logistic
from Functions import *

def test_Logistic_simple():
    """
    Test that logistic regression can fit 0<x<0.5 to y = 0 and 1>x>0.5 to y=1
    (should get at least 90% accuracy)
    """
    L = Logistic()
    N_train = 1000
    N_test = 1000
    x_train = np.random.random(N_train)[:,None]
    y_train = np.round(x_train).flatten()
    x_test = np.random.random(N_test)[:,None]
    y_test = np.round(x_test).flatten()
    L.fit(x_train,y_train,N = 1000,eta = 0.1)
    preds = np.round(L.predict(x_test)).flatten()
    acc = np.sum(preds == y_test)/N_test
    assert acc > 0.9, "Logistic Regression failed a simple classification task, something is probably wrong"

def test_NN_simple():
    """
    Test that the neural network can fit 0<x<0.5 to y = 0 and 1>x>0.5 to y=1
    (should get at least 90% accuracy)
    """
    NNmodel = FFNN(hlayers = [20,10], activation = ReLU(0.01), outactivation = Softmax(), cost = CrossEntropy(), Xfeatures = 1, yfeatures = 2, eta = 0.1)
    N_train = 1000
    N_test = 1000
    x_train = np.random.random(N_train)[:,None]
    y_train = np.array([np.round(x_train).flatten(),1-np.round(x_train).flatten()]).T
    x_test = np.random.random(N_test)[:,None]
    y_test = np.round(x_test).flatten()
    NNmodel.fit(x_train,y_train,n_epochs = 1000)
    preds = np.round(NNmodel.predict(x_test)[0,:])
    acc = np.sum(preds == y_test)/N_test
    print(preds.shape)
    assert acc > 0.9, "Neural Net failed a simple classification task, something is probably wrong"