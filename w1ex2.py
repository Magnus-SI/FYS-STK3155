import numpy as np
import sklearn.linear_model as sklm
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
np.random.seed(3)
x=np.random.rand(100)
y=5*x**2 + 0.1*np.random.randn(100)

"Without scikit-learn"
X=np.zeros((len(x),3))
X[:,0]=1; X[:,1]=x; X[:,2]=x**2
beta=np.einsum('ij,kj,k', np.linalg.inv(np.einsum('ij,ik',X,X)), X, y)
xsort=np.sort(x)
fit2=beta[0]+beta[1]*xsort+beta[2]*xsort**2
#other=np.linalg.lstsq(X,y,rcond=None)[0]
#otherfit=other[0]+other[1]*xsort+other[2]*xsort**2
plt.plot(x,y,"r.",label="Data")
plt.plot(xsort,fit2,label="2nd order poly fit")
#plt.plot(xsort,otherfit,label="other")
#plt.legend()
#plt.show()

"With scikit-learn"
solve=sklm.LinearRegression(False)      #intercept already in X[0]
solve.fit(X,y)
sklfit=solve.coef_[0]+solve.coef_[1]*xsort+solve.coef_[2]*xsort**2
plt.plot(xsort,sklfit,label="sklfit")
plt.legend()
plt.show()

"Error analysis:"
fitMSE=MSE(y,solve.predict(X))
fitR2=solve.score(X,y)
print("\nMean Square Error: %g"%fitMSE)
print("R2 Value: %g"%fitR2)
