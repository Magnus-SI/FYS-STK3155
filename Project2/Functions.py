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
        return x-target

class Accuracy:
    """
    Should not be used as an actual cost function, since it has no good derivative
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
    def __init__(self,threshold):
        self.threshold = threshold

    def __call__(self,x,target):
        if len(target.shape) == 2:
            y = target.flatten()
        else:
            y = target
        if len(x.shape) == 2:
            x = x.flatten()>self.threshold
        else:
            x = x>self.threshold
        return confusion_matrix(y,x).ravel()

class CmatNN:
    """
    Uses scikit learns confusion matrix
    returns array with TN, FP, FN, TP for binary
    takes two vectors, where the first is the probability for 1
    """
    def __init__(self,threshold):
        self.threshold = threshold

    def __call__(self,x,target):
        y = target[:,0]
        x = x[0,:]>self.threshold
        return confusion_matrix(y,x).ravel()




class Accu2:
    def __call__(self, x, target):
        y = target.T
        return np.count_nonzero((np.round(x)-y) == 0, axis=1)/y.shape[1]

class ELU:
    def __init__(self,a):
        self.a = a

    def __call__(self,x):
        return x*(x>0) + self.a*(np.exp(x)-1)*(x<0)

    def derivative(self,x):
        return (x>0) + self.a*np.exp(x)*(x<0)
