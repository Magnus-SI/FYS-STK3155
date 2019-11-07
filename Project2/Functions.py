import numpy as np
from sklearn.metrics import confusion_matrix

class ReLU:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return x*(x>0) + self.a*x*(x<=0)

    def derivative(self, x):
        return 1.0*(x>0) + self.a*(x<=0)

class Softmax:
    def __call__(self,x):
        return np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)

    def derivative(self,x):
        out = self.__call__(x)
        return out*(1-out)

class MSE:
    """
    Cost function, so it takes two arguments, value and target
    """
    def __call__(self,x,target):
        return 0.5*np.sum((x-target)**2)

    def derivative(self,x,target):
        return x-target

class Sigmoid:
    def __call__(self,x):
        return 1/(1+np.exp(-x))

    def derivative(self,x):
        out = self.__call__(x)
        return out*(1-out)


class CrossEntropy:
    def __call__(self,x,target):
        return -np.sum(target*np.log(x) + (1-target)*np.log(1-x))

    def derivative(self,x,target):
        return (x-target)/(x*(1-x))

class Accuracy:
    """
    Should not be used as an actual cost function,since it has no good derivative
    """
    def __call__(self,x,target):
        if len(target.shape) == 2:
            y = target[:,0]
        elif len(y.shape) == 1:
            y = target
        return np.count_nonzero(np.round(x)==y)/len(y)

class FalseRate:
    def __call__(self, x, target):
        if len(target.shape) == 2:
            y = target[:,0]
            if len(x.shape) == 2:
                x = x[0]
        elif len(y.shape) == 1:
            y = target
            #x = x[0]
        x = (x>0.5)*1
        false_negative = np.sum((x-y)==-1)
        false_positive = np.sum((x-y)==1)
        true_positive = np.sum((x==1) * (y==1))
        true_negative = np.sum((x==0) * (y==0))
        print(false_negative, true_positive,  false_positive,  true_negative)
        falseposrate = false_positive/(false_positive + true_negative)
        falsenegrate = false_negative/(false_negative + true_positive)
        return np.array([falseposrate, falsenegrate])

class Cmat:
    """
    Uses scikit learns confusion matrix
    returns array with TN, FP, FN, TP for binary
    """
    def __call__(self,x,target):
        if len(target.shape) == 2:
            y = target.flatten()
        else:
            y = target
        if len(x.shape) == 2:
            x = np.round(x.flatten())
        else:
            x = np.round(x)
        return confusion_matrix(y,x).ravel()

class Cmat_with_ignore:
    """
    Uses scikit learns confusion matrix
    Can ignore all results within 50 +/- an interval (sigma)
    (Treats these as too uncertain to classify)
    sigma should be between 0 and 0.5
    returns array with TN, FP, FN, TP and number of ignored datapoints
    """
    def __init__(self,sigma):
        self.lb = 0.5-sigma # lower bound
        self.ub = 0.5+sigma # upper bound

    def __call__(self,x,target):
        if len(target.shape) == 2:
            y = target.flatten()
        else:
            y = target
        if len(x.shape) == 2:
            x = x.flatten()
        true_arr = (x<self.lb) | (x>self.ub)
        n_ignored = true_arr.size - np.count_nonzero(true_arr)
        y = y[true_arr]
        x = np.round(x[true_arr])
        cmat = confusion_matrix(y,x).ravel()
        return_arr = np.zeros(cmat.size+1)
        return_arr[:-1] = cmat
        return_arr[-1] = n_ignored
        return return_arr


class Accu2:
    def __call__(self, x, target):
        y = target.T
        #print(x, y)
        #import yyoeror
        return np.count_nonzero((np.round(x)-y) == 0, axis=1)/y.shape[1]
