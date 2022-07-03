# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:31:30 2022

@author: Şiyar Öner
"""

#importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import seaborn as sns

#importing data and slicing
data=pd.read_csv("kc_house_data.csv")
# print(data.dtypes)
# print(data.isnull().any())
# print(data.groupby(["zipcode"]).count())
# print(data.corr().sort_values("price")["price"])
# with sns.plotting_context("notebook",font_scale=3):
#     g = sns.pairplot(data[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
#                  hue='bedrooms', palette='tab20',size=6)
# g.set(xticklabels=[]);
y=data["price"]
x=data.drop(columns=["id","zipcode","date","price"])

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.33)

#fit
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print(r2_score(y_test,pred))

#Backward elemination step 1

X=np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)
X=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
model=sm.OLS(y, X).fit()
summary=model.summary()
print(model.pvalues.astype(float).sort_values()>0.05)

#Backward elemination 2

X=np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)
X=X[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]]
model=sm.OLS(y, X).fit()
summary=model.summary()
print(model.pvalues.astype(float).sort_values()>0.05)