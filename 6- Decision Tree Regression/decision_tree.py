# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:44:15 2022

@author: Şiyar Öner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeRegressor


#Importing Data and slicing
data=pd.read_csv("maaslar.csv")
x=data.iloc[:,1]
y=data.iloc[:,2]
x=np.array(x).reshape(-1,1)

#decision tree
dt=DecisionTreeRegressor(random_state=0)
dt.fit(x,y)
pred=dt.predict(x)
plt.scatter(x,y,color="red")
plt.plot(x,pred)
plt.show()
x1=x+0.5
x2=x-0.4
print(dt.predict([[6.6]]))
print(dt.predict([[11]]))

plt.title("x1")

pred1=dt.predict(x1)
plt.scatter(x,y,color="red")
plt.plot(x,pred1)
plt.show()
plt.title("x2")
pred2=dt.predict(x2)
plt.scatter(x,y,color="red")
plt.plot(x,pred2)
