# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 18:17:55 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



#import data and slicing
data=pd.read_csv("data.csv")
height=data["height"]
countries=data["country"]
gender=data["gender"]
wa=data[["weight","age"]]

#label and onehot encoder
le=LabelEncoder()
ohe=OneHotEncoder()
gender=le.fit_transform(gender)
countries=(le.fit_transform(countries)).reshape(-1,1)
countries=ohe.fit_transform(countries).toarray()

#concatenating data
x=pd.concat([pd.DataFrame(data=countries,columns=["fr","tr","us"]),wa,pd.DataFrame(gender,columns=["gender"])],axis=1)
#x=x.drop(columns="age") #to increase accuracy

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,height,test_size=0.33,random_state=1)

lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)

r2=r2_score(y_test,pred)
print(r2)

# Backward Elimination
import statsmodels.api as sm

# Adding B0 with ones matrix
X=np.append(arr=np.ones(((len(x)),1)).astype(int),values=x,axis=1)
Xbw=X[:,[0,1,2,3,4,5,6]]
Xbw=np.array(Xbw,dtype=float)
model=sm.OLS(height,Xbw).fit()
print(model.summary())


#eleminate column 5
X=np.append(arr=np.ones(((len(x)),1)).astype(int),values=x,axis=1)
Xbw=X[:,[0,1,2,3,4,6]]
Xbw=np.array(Xbw,dtype=float)
model=sm.OLS(height,Xbw).fit()
print(model.summary())









