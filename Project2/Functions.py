import numpy as np

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
