import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as R2
import random
import time as tm
import sys



def predict(X,w,b):
    y_pred = np.dot(w, X.T) + b
    return y_pred

def mini_batch_gradient_descent(epochs = 100, batch_size = 5, learning_rate = 0.01):
   
    number_of_features = d-1
    # numpy array with 1 row and columns equal to number of features. In
    # our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(number_of_features))
    b = 0
    cost_list = []
    epoch_list = []
    r2_list= []
    for i in range(epochs):
        print(f'started {i}')
        t1 = tm.time()
        for df in pd.read_csv(ds, chunksize=batch_size,usecols=np.arange(d),header=None):

            Xj = np.array(df.iloc[:,1:])
            yj = np.array(df.iloc[:,0])
            y_predicted = np.dot(w, Xj.T) + b

            w_grad = -(2/len(Xj))*(Xj.T.dot(yj-y_predicted))
            b_grad = -(2/len(Xj))*np.sum(yj-y_predicted)

            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad

            cost = np.mean(np.square(yj-y_predicted)) # MSE (Mean Squared Error)

       
                 
        print(f'epoch {i} tot: {tm.time()-t1} cost: {cost}')
        cost_list.append(cost)
        epoch_list.append(i)
       
    return w, b, cost, cost_list, epoch_list
ds = sys.argv[1]
d = int(sys.argv[2])
n = int(sys.argv[3])
e = int(sys.argv[4])
bs = int(sys.argv[5])
lr = float(sys.argv[6])



s = tm.time()
w, b, cost, cost_list, epoch_list = mini_batch_gradient_descent(
    epochs = e,
    batch_size = bs,
    learning_rate = lr
)
tot = tm.time() - s
test_size = int(n*.3)

df = pd.read_csv(ds,nrows=test_size,usecols=np.arange(d),header=None)
scaled_y = np.array(df.iloc[:,0])
scaled_X = np.array(df.iloc[:,1:])


r2 = R2(scaled_y,predict(scaled_X,w,b))

out = ['sgd',ds,tot,r2,e,bs,lr]
out = pd.DataFrame([out])
out.to_csv('gammasgdresults_3.csv',index=False,header=None,mode='a')
