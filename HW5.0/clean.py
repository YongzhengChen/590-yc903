#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 20:01:21 2021

@author: chenyongzheng
"""

import pandas as pd

with open('Holmes.txt') as f:
    contents = f.readlines()
    
with open('OliverTwist.txt') as f:
    contents1 = f.readlines()
    
with open('Twilight.txt') as f:
    contents2 = f.readlines()
    
    
df = pd.DataFrame()

i = 0
while i <= 19:
    strings1 = ''
    for j in range(i*len(contents)//20,(i+1)*(len(contents)//20)):
        strings1 = strings1 + ' ' + contents[j]
    next_dict = {'contexts' : [strings1], 'Novel': ['Holmes']}
    next_df = pd.DataFrame(next_dict)
    df = pd.concat([df,next_df],axis = 0)
    
    strings2 = ''
    for j in range(i*len(contents1)//20,(i+1)*(len(contents1)//20)):
        strings2 = strings2 + ' ' + contents1[j]
    next_dict1 = {'contexts' : [strings2], 'Novel': ['OliverTwist']}
    next_df1 = pd.DataFrame(next_dict1)
    df = pd.concat([df,next_df1],axis = 0)
    
    strings3 = ''
    for j in range(i*len(contents2)//20,(i+1)*(len(contents2)//20)):
        strings3 = strings3 + ' ' + contents2[j]
    next_dict2 = {'contexts' : [strings3], 'Novel': ['Twilight']}
    next_df2 = pd.DataFrame(next_dict2)
    df = pd.concat([df,next_df2],axis = 0)
    
    i += 1
    
df = df.reset_index(drop = True)
    

df.to_csv('Novel.csv',header=True)
    
        