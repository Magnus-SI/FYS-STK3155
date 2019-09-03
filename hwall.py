import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

class hw:
    def __init__(self,N=100):
        self.N=N

    def gendata(self,seed=3):
        np.random.seed(seed)
        self.x=np.random.rand(self.N)
        x=self.x
        self.yactual=5*x**2+0.1
        self.ynoise=self.yactual+0.1*np.random.randn(self.N)

        X=np.zeros((len(x),3))
        X[:,0]=1; X[:,1]=x; X[:,2]=x**2
        self.X=X

    def OLinReg(self):
        x=self.x
        X=self.X
        y=self.ynoise
        XTXinv=np.linalg.inv(np.einsum('ij,ik',X,X))
        var=np.diagonal(XTXinv)
        beta=np.einsum('ij,kj,k', XTXinv, X, y)
        fit=beta[0]+beta[1]*x+beta[2]*x**2
        return fit, var

    def SLinReg(self):
        x=self.x
        X=self.X
        y=self.ynoise
        lr=LinearRegression(fit_intercept=False)
        lr.fit(X,y)
        beta=lr.coef_
        fit=beta[0]+beta[1]*x+beta[2]*x**2
        return fit

    def ORidge(self,lambd):
        x=self.x; X=self.X; y=self.ynoise
        XTX=np.einsum('ij,ik',X,X)      #X^T X
        XTX+=lambd*np.identity(len(X[0]))
        XTXinv=np.linalg.inv(XTX)
        var=np.diagonal(XTXinv)
        beta=np.einsum('ij,kj,k', XTXinv, X, y)
        fitridge=beta[0]+beta[1]*x+beta[2]*x**2
        return fitridge, var

    def SRidge(self,lambd):
        x=self.x; X=self.X; y=self.ynoise
        lr=Ridge(lambd,fit_intercept=False)
        lr.fit(X,y)
        beta=lr.coef_
        fitridge=beta[0]+beta[1]*x+beta[2]*x**2
        return fitridge

    def Error(self,fit):
        errnoise=MSE(self.ynoise, fit)
        erractual=MSE(self.yactual, fit)
        return errnoise, erractual

    def multicomperror(self, lambds, noises):
        MSEs=np.zeros((len(noises),len(lambds)+1))
        i=0
        vars=np.zeros((len(noises), len(lambds)+1,3))
        for noise in noises:
            self.ynoise=self.yactual+noise*np.random.randn(self.N)
            lrfit,var=self.OLinReg()
            MSEs[i,0]=self.Error(lrfit)[1]
            vars[i,0]=var
            j=1
            for lambd in lambds:
                rfit,var=self.ORidge(lambd)
                MSEs[i,j]=self.Error(rfit)[1]
                vars[i,j]=var
                j+=1
            i+=1
        lambds0=np.zeros(len(lambds)+1)
        lambds0[1:]=lambds
        return MSEs, lambds0, vars

    def plotMSEcomp(self,lambds,noises):
        MSEs, lambds0, vars = self.multicomperror(lambds,noises)
        for i in range(len(noises)):
            plt.plot(lambds0,MSEs[i], label="sigma = %.1e"%noises[i])
        plt.xlabel("lambda")
        plt.ylabel("MSEs")
        plt.xscale("log")
        plt.yscale("log")
        #plt.yscale("log")

        plt.legend()
        plt.show()

    def plotvars(self,lambds,noises):
        MSEs, lambds0, vars = self.multicomperror(lambds,noises)
        for i in range(len(noises)):
            plt.plot(lambds0,vars[i,:,1],label="sigma = %.1e"%noises[i])
        plt.xlabel("lambda")
        plt.ylabel("Beta1 variance")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

if __name__=="__main__":
    one=hw(100)
    one.gendata(seed=13)
    lambds=np.logspace(-10,0,100)#np.array([0,1e-4,1e-3,1e-2,1e-1,1])
    noises=np.linspace(0.05,1,5)#np.array([0.05,0.1,0.15,0.2])
    #one.plotMSEcomp(lambds,noises)
    "Ikke uventet at MSE generelt er bedre for linear regression siden dette skal optimalisere MSE"
    noises=np.linspace(0.05,0.15,2)
    lambds=np.logspace(-5,5,100)
    one.plotvars(lambds,noises)
