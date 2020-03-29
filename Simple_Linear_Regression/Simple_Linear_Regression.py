# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=1/3,random_state=0)

#implement our classifier based on simple linear regression

from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(x_train,y_train)

y_predict=simplelinearRegression.predict(x_test)

#implement the Graph
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,simplelinearRegression.predict(x_train))
plt.show()

