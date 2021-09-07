#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:55:43 2021

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
    
    def age_18(self,df):
        df = df[df['x']<18]
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
df_unno = preprocess.age_18(df)
age = df_unno['x'].tolist()
mean_x, mean_y, std_x, std_y, df = preprocess.normalize(df_unno)
train, test = preprocess.split(df)
train = train.reset_index(drop = True)

def linear_regression_mse(para):
    w = para[0]
    b = para[1]
    global train
    mse = 0
    for i in range(len(train)):
        mse+=(w*train['x'][i]+b-train['y'][i])**2
    return mse

res = minimize(linear_regression_mse,[0,0],method='Nelder-Mead',tol = 1e-5)
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
            res_y = res_x*parameters[0]+parameters[1]   ## unnormalize
            uno_y = res_y*std_y+mean_y
            y.append(uno_y)
    else:
        res_x = (x-mean_x)/std_x
        res_y = res_x*parameters[0]+parameters[1]
        y = res_y*std_y+mean_y
    return y

predictions = predict(age)

plt.plot(age,predictions)
plt.scatter(df_unno['x']*std_x+mean_x,df_unno['y']*std_y+mean_y)

plt.show()



        
        
        
            
        
    
        
        