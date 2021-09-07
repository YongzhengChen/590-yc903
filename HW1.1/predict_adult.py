#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:30:12 2021

@author: chenyongzheng
"""

import pandas as pd
import numpy as np
from   scipy.optimize import minimize
import matplotlib.pyplot as plt

class data():
  
    
    def __init__(self):
        return
    
    def load(self,filename):
        df = pd.read_json(filename)
        return df
    
    def split(self,df):
        train = pd.DataFrame()
        test  = pd.DataFrame()
        for i in range(len(df)):
            if i%5 == 0:
                test = test.append(df.loc[i,])
            else:
                train = train.append(df.loc[i,])
        return train, test
    
preprocess = data()
df = preprocess.load('weight.json')    
weight = df['y'].tolist()
train, test = preprocess.split(df)
train = train.reset_index(drop = True)

def logistic_regression(para):
    A  = para[0]
    w  = para[1]
    x0 = para[2]
    s  = para[3]
    global train
    mse = 0
    for i in range(len(train)):
        mse += ((A/(1+np.e**((x0-train['y'][i])/w)))+s-train['is_adult'][i])**2
    return mse

res = minimize(logistic_regression,[0,0,0,0],method='Powell',tol = 1e-5)
parameters = res.x

def predict(x):
    global parameters
    if type(x) == list:
        y = []
        for i in range(len(x)):
            res_y = parameters[0]/(1+np.e**((parameters[2]-x[i])/parameters[1]))+parameters[3]   ## unnormalize
            y.append(res_y)
    else:
        y = parameters[0]/(1+np.e**((parameters[2]-x)/parameters[1]))+parameters[3]
    return y

predictions = predict(weight)

plt.plot(weight,predictions,color = 'red')
plt.scatter(df['y'],df['is_adult'])

plt.show()