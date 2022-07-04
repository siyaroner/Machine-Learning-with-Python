# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:21:19 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm


#import data slicing
data=pd.read_csv("fish.csv")
col=data.columns
# print(data.isnull().any())
# print(data.dtypes)
# print(data.describe().head(20))
# print(data.corr())
y=data["Weight"]
x=data.drop(columns=["Weight"])
x=pd.get_dummies(x)

#Bacward Elemination step 1
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

#Bacward Elemination step 2 removing first max value whichs is 5>0.05
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,3,4,6,7,8,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

#Bacward Elemination step 3 removing first max value whichs is 5>0.05
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,4,6,8,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

#train test split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.33)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
y_train=np.array(y_train).reshape(-1,1)
y_train=sc.fit_transform(y_train)

#svr
svr=SVR(kernel="rbf")
svr.fit(x_train,y_train)
pred=svr.predict(x_test)
pred=sc.inverse_transform(pred.reshape(-1,1)).reshape(-1)
r2=r2_score(y_test,pred)
print(r2)
