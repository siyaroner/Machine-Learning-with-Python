# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 18:18:01 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statsmodels.api as sm

#import data,analysis and slicing
data=pd.read_csv("fish.csv")
# print(data.isnull().any())
# print(data.dtypes)
# dummies=pd.get_dummies(data)
# print(dummies.corr()["Weight"])
y=data["Weight"]
df=data.drop(columns=["Weight"])
x=pd.get_dummies(df)

# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

#fit with linear regression
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
r2=r2_score(y_test,pred)

# Backward Elimintaion step 1
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

# Backward Elimintaion step 2
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,3,4,6,7,8,9,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

# Backward Elimintaion step 3
X=np.append(arr=np.ones((len(x),1)).astype(float),values=x,axis=1)
X=X[:,[0,1,2,3,4,6,7,8,10,11]]
model=sm.OLS(y,X).fit()
print(model.summary())

#Polynomial Regression
pl=PolynomialFeatures(10)
X_poly=pl.fit_transform(X)
lr2=LinearRegression()
lr2.fit(X_poly,y)
plt.title("Degree 10")
plt.scatter(x.iloc[:,0].values,y)
a=lr2.predict(X_poly)
plt.plot(x.iloc[:,:3].values,a)
# plt.show()






