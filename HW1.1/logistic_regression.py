#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:12:27 2021

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
    def normalize(self,df):
        mean_x = sum(df['x'])/len(df)
        mean_y = sum(df['y'])/len(df)
        std_x  = np.std(df['x'])
        std_y  = np.std(df['y'])
        for i in range(len(df)):
            df['x'][i] = (df['x'][i]-mean_x)/std_x
            df['y'][i] = (df['y'][i]-mean_y)/std_y
        return mean_x, mean_y, std_x, std_y, df
    
preprocess = data()
df = preprocess.load('weight.json')    
age = df['x'].tolist()
mean_x, mean_y, std_x, std_y, df = preprocess.normalize(df)
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
        mse += ((A/(1+np.e**((x0-train['x'][i])/w)))+s-train['y'][i])**2
    return mse

res = minimize(logistic_regression,[0,0,0,0],method='Nelder-Mead',tol = 1e-5)
parameters = res.x

def predict(x):
    global mean_x
    global mean_y
    global std_x
    global std_y
    global parameters
    if type(x) == list:
        y = []
        for i in range(len(x)):
            res_x = (x[i]-mean_x)/std_x
            res_y = parameters[0]/(1+np.e**((parameters[2]-res_x)/parameters[1]))+parameters[3]   ## unnormalize
            uno_y = res_y*std_y+mean_y
            y.append(uno_y)
    else:
        res_x = (x-mean_x)/std_x
        res_y = parameters[0]/(1+np.e**((parameters[2]-res_x)/parameters[1]))+parameters[3]
        y = res_y*std_y+mean_y
    return y

predictions = predict(age)

plt.plot(age,predictions,color = 'red')
plt.scatter(df['x']*std_x+mean_x,df['y']*std_y+mean_y)

plt.show()

