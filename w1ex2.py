import numpy as np
import sklearn.linear_model as sklm
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
np.random.seed(3)
x=np.random.rand(100)
y=5*x**2 + 0.1*np.random.randn(100)
X=np.zeros((len(x),3))
X[:,0]=1; X[:,1]=x; X[:,2]=x**2
xsort=np.sort(x)

def Linregown():
    "Without scikit-learn"

    beta=np.einsum('ij,kj,k', np.linalg.inv(np.einsum('ij,ik',X,X)), X, y)

    fit2=beta[0]+beta[1]*xsort+beta[2]*xsort**2
    #other=np.linalg.lstsq(X,y,rcond=None)[0]
    #otherfit=other[0]+other[1]*xsort+other[2]*xsort**2
    plt.plot(x,y,"r.",label="Data")
    plt.plot(xsort,fit2,label="LinReg fit")
    #plt.plot(xsort,otherfit,label="other")
    #plt.legend()
    #plt.show()
    return fit2

"With scikit-learn"
def Linregsci():
    solve=sklm.LinearRegression(False)      #intercept already in X[0]
    solve.fit(X,y)
    sklfit=solve.coef_[0]+solve.coef_[1]*xsort+solve.coef_[2]*xsort**2
    plt.plot(xsort,sklfit,label="Linreg fit")

    "Error analysis:"
    fitMSE=MSE(y,solve.predict(X))
    fitR2=solve.score(X,y)
    print("\nMean Square Error: %g"%fitMSE)
    print("R2 Value: %g"%fitR2)

    return sklfit



def Ridgeplot(lambd):
    "Week 2 ex 4:"
    XTX=np.einsum('ij,ik',X,X)      #X^T X
    XTX+=lambd*np.identity(len(X[0]))
    XTXinv=np.linalg.inv(XTX)
    beta=np.einsum('ij,kj,k', XTXinv, X, y)
    fitridge=beta[0]+beta[1]*xsort+beta[2]*xsort**2
    #plt.plot(x,y,"r.",label="Data")
    plt.plot(xsort,fitridge,label="Ridge fit, lamd=%.1e"%lambd)
    return fitridge

def Ridgescikit(lambd):
    lr=sklm.Ridge(lambd,fit_intercept=False)
    lr.fit(X,y)
    beta=lr.coef_
    fitridge=beta[0]+beta[1]*xsort+beta[2]*xsort**2
    plt.plot(xsort,fitridge,label="scikit Ridge, lambd=%.1e"%lambd)
    return fitridge

if __name__=="__main__":
    lrfit=Linregown()
    for lambd in (0,0.1,0.01):#,0.001,0.0001):
        ro=Ridgeplot(lambd)
        rs=Ridgescikit(lambd)
    plt.legend()
    plt.show()

    pritn("SumDiff: %g"%(ro-rs))
