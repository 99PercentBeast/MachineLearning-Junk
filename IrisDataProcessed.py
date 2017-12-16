#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:43:01 2017

@author: aasim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Dataset Imported
and Splitted into 2 
X =( sepal_length,sepal_width,petal_length,petal_width)
y = species
'''
dataset=pd.read_csv('iris.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4:5].values.ravel()
'''
Taking care of missing Data
'''

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
X=imputer.fit_transform(X)

'''
Encoding the data with OneHotEncoder and LabelEncoder
sesota = 0
versicolor = 1
virginica=2
'''
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
'''
If data will be in X we would be using OneHotEncoder
onehotencoder_X = OneHotEncoder(categorical_feature=[0])
X = onehotencoder_X.fit_transform(X).toarray()
'''


'''
Training Set and Testing Set
'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state = 42)

'''
feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)





