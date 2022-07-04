# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:37:02 2022

@author: Şiyar Öner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor

#Importing Data and slicing
data=pd.read_csv("maaslar.csv")
x=data.iloc[:,1]
y=data.iloc[:,2]
x=np.array(x).reshape(-1,1)
x1=x+0.5
x2=x-0.4

# random forest regressor
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(x,y)
print(rfr.predict([[6.6]]))

plt.scatter(x,y,color="red")
plt.plot(x,rfr.predict(x),color="blue")

plt.plot(x,rfr.predict(x1),color="green")
plt.plot(x,rfr.predict(x2),color="yellow")