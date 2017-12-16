#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:41:44 2017

@author: aasim
"""

'''
Multiple Regression
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values.ravel()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lencode = LabelEncoder()
y=lencode.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)

import statsmodels.formula.api as sm
'''
Backward Propogation
'''
X = np.append(arr=np.ones(150,1).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]




