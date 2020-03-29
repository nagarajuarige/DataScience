# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:12:29 2020

@author: Nagaraju
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x))
plt.show()

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid))
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()