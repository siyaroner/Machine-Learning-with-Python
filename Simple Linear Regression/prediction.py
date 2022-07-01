# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:20:57 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# reading data and slicing
data=pd.read_csv("sales_data.csv")
months=pd.DataFrame(data["Aylar"])
print(type(months),months)
sales=pd.DataFrame(data["Satislar"])
print(type(sales),sales)

# split,train,test
x_train,x_test,y_train,y_test=train_test_split(months,sales,test_size=0.30,random_state=10)
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print("pred",pred)
print(y_test)
# x_train=x_train.sort_index()
# y_train=y_train.sort_index()
# plt.scatter(x_train,y_train)
x_tests=x_test.sort_index()
y_test=y_test.sort_index()
plt.scatter(x_tests,y_test)
plt.plot(x_test.values,pred,"r")
