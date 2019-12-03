import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading data
df = pd.read_csv("pulsar_stars.csv")
data = df.values[:,:-1]
labels = df.values[:,-1]
indices = np.arange(data.shape[0])

train_size = 0.8
train_inds = np.random.choice([True,False],size=len(indices), replace=True, p = [train_size,1-train_size])
X_train = data[train_inds]; X_test = data[~train_inds]
y_train = labels[train_inds]; y_test = labels[~train_inds]

# Testing XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 2
param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)
ypred_XGB = bst.predict(dtest)

#Using NN for comparison
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(50, activation = 'relu'),
        tf.keras.layers.Dense(2, activation = 'softmax')
    ]
)

# Compile model
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['AUC']
)
# Fit to training data
y_train_2d = np.zeros((len(y_train),2))
y_test_2d = np.zeros((len(y_test),2))

y_train_2d[:,0] = y_train; y_train_2d[:,1] = 1-y_train
y_test_2d[:,0] = y_test; y_test_2d[:,1] = 1-y_test
model.fit(
    X_train,
    y_train_2d,
    epochs = 100,
    batch_size = 32,
    validation_data = (X_test,y_test_2d)
)

ypred_NN = model.predict(X_test)

plt.figure(figsize=(7,5))
fpr, tpr, thres = metrics.roc_curve(y_test, ypred_XGB)
plt.plot(fpr,tpr,label = 'XGB')
fpr, tpr, thres = metrics.roc_curve(y_test, ypred_NN[:,0])
plt.plot(fpr,tpr)
plt.plot(fpr,tpr,label = 'NN')
plt.legend()
plt.show()