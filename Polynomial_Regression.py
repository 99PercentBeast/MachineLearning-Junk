#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:03:28 2017

@author: aasim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1].values.reshape(-1,1)
y = dataset.iloc[:,2].values.reshape(-1,1)
'''
No need becoz data is not big
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
'''

#LinearRegressionModel
from sklearn.linear_model import LinearRegression
linearReg = LinearRegression()
linearReg.fit(X,y)

#LinearModel Plotting or visualizing data
plt.scatter(X,y,color='blue')
plt.plot(X,linearReg.predict(X),color='red')
plt.title('LinearRegression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
polyReg=PolynomialFeatures(degree = 5)
X_poly = polyReg.fit_transform(X)

#Create another LinearRegression Object
linearReg2=LinearRegression()
linearReg2.fit(X_poly,y)


#for better plotting use X_grid
X_grid = np.arange(min(X),max(X),0.01).reshape(-1,1)


#Polynomial Model Plotting or visualizing Polynomial data
plt.scatter(X,y,color='blue')
plt.plot(X_grid,linearReg2.predict(polyReg.fit_transform(X_grid)),color='red')
plt.title('Polynomial Linear Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
