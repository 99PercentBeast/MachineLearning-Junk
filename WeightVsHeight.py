#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:14:19 2017
@author: aasim
"""
'''
Importing Libraries

numpy -> arrays processing
matplotlib -> plotting graph
pandas -> to read files // Not used in this module

sklearn -> aka SciKitLearn
sklearn provides us with some awesome Machine Learning Modules
linear_model -> is one of the module which contains our class LinearRegression

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
assinging some values for weight and height and convert them to arrays
using np.array() function
using array.reshape(-1,1) to shape my array into 2D array
becoz SimpleLinearRegression is done on 2D models0

'''
weight = [127,121,142,157,162,156,169,165,181,208]
height = [63,64,66,69,69,71,71,72,73,75]
weight = np.array(weight).reshape(-1,1)
height = np.array(height).reshape(-1,1)
'''
Simple Linear Regression
Weight vs Height
'''
from sklearn.linear_model import LinearRegression
'''
regressor = creating object for our class LinearRegression()
'''
regressor = LinearRegression()
'''
regressor.fit() = Fit our linear model
'''
regressor.fit(height,weight)

'''
Predicting our value ___  and saving in pred variable;
'''
predict_my_weight =float( input('Enter your height : '))
predicted_weight = regressor.predict(predict_my_weight)
'''
Plotting our graph
'''
plt.scatter(height,weight,color='red')
plt.plot(height,regressor.predict(height),color='blue')
plt.title('Height vs Weight Simple Linear Regression')
plt.xlabel('height')
plt.ylabel('weight')
plt.plot()

print(predicted_weight)