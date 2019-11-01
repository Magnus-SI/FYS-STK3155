import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    df = pd.read_excel('ccdefaults.xls', skiprows = [1])
    #Set correct data limits:
    Ts = np.ones((23,len(df)))
    Ts[1] = (df['X2'].values == 1) + (df['X2'].values == 2)
    Ts[2] = (df['X3'].values == 1) + (df['X3'].values == 2)\
        +(df['X3'].values == 3) + (df['X2'].values == 4)
    Ts[3] = (df['X4'].values == 1) + (df['X4'].values == 2)\
        +(df['X4'].values == 3)
    for i in range(6,12):
        t = np.zeros(len(df))>np.ones(len(df))
        for j in range(-1,10):
            t+=df['X%i'%i].values == j
        Ts[i-1] = t
    T = np.prod(Ts, axis=0)
    df = df[T.astype(bool)]

    X_vars = df.keys()[1:-1]
    X = df[X_vars].values
    corr=np.corrcoef(X.T).round(2)
    sns.heatmap(data = corr, annot = True, cmap = 'viridis', xticklabels = X_vars, yticklabels = X_vars)
