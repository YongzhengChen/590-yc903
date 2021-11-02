#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:40:56 2021

@author: chenyongzheng
"""

from keras.models import load_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import keras


model = keras.models.load_model('1dcnn_model.h5')

print(model.summary())

df = pd.read_csv('test.csv')

X_data=df['contexts']
y_labels=df['Novel']
X_data_train,X_data_test,y_labels_train,y_labels_test=train_test_split(X_data,y_labels,test_size=0.2,random_state=1000)

tfidf_vectorizer=TfidfVectorizer()

train_tfidf=tfidf_vectorizer.fit_transform(X_data_train)

test_tfidf=tfidf_vectorizer.transform(X_data_test)

y_labels_test = y_labels_test.reset_index(drop=True)
y_labels_train = y_labels_train.reset_index(drop=True)

label_dic = {'Holmes':0,'OliverTwist':1,'Twilight':3}
for i in range(len(y_labels_train)):
    y_labels_train[i]= label_dic[y_labels_train[i]]
    
for i in range(len(y_labels_test)):
    y_labels_test[i]= label_dic[y_labels_test[i]]

score = model.evaluate(train_tfidf.toarray().astype('float32'), y_labels_train.astype('float32'), verbose=0)

print("%s: %.2f%%" % ('training'+model.metrics_names[1], score[1]*100))


score = model.evaluate(test_tfidf.toarray().astype('float32'), y_labels_test.astype('float32'), verbose=0)

print("%s: %.2f%%" % ('test'+model.metrics_names[1], score[1]*100))

model = keras.models.load_model('simple_model.h5')

score = model.evaluate(train_tfidf.toarray().astype('float32'), y_labels_train.astype('float32'), verbose=0)

print("%s: %.2f%%" % ('training'+model.metrics_names[1], score[1]*100))


score = model.evaluate(test_tfidf.toarray().astype('float32'), y_labels_test.astype('float32'), verbose=0)

print("%s: %.2f%%" % ('test'+model.metrics_names[1], score[1]*100))


