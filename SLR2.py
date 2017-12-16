#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:14:19 2017
@author: aasim
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

weight = [127,121,142,157,162,156,169,165,181,208]
height = [63,64,66,69,69,71,71,72,73,75]
weight = np.array(weight).reshape(-1,1)
height = np.array(height).reshape(-1,1)
'''
Simple Linear Regression
Weight vs Height
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(weight,height)
pred = regressor.predict(143)

plt.scatter(weight,height,color='red')
plt.plot(weight,regressor.predict(weight),color='blue')
plt.title('Weight vs Height Simple Linear Regression')
plt.xlabel('weight')
plt.ylabel('height')
plt.plot()