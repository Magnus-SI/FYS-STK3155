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
        out = self.__call__(self,x)
        return out*(1-out)

class MSE:
    def __init__(self,target):
        self.target = target

    def __call__(self,x):
        return 0.5*np.sum((x-self.target)**2)

    def derivative(self,x):
        return x-self.target

class sigmoid:
    def __call__(self,x):
        return 1/(1+np.exp(-x))

    def derivative(self,x):
        out = self.__call__(x)
        return out*(1-out)

