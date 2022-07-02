# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:26:01 2022

@author: Şiyar Öner
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

#import data
data=pd.read_csv("tennis.csv")
play=data.iloc[:,-1]
temp_hum=data.iloc[:,1:3]

# get dummies
outlook_dummies=pd.get_dummies(data["outlook"])
y=pd.get_dummies(play).iloc[:,-1]
windy=pd.get_dummies(data["windy"]).iloc[:,-1]
#concatenating data
x=pd.concat((outlook_dummies,temp_hum,windy),axis=1)

# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

#train
lr=LinearRegression()

lr.fit(x_train,y_train)
pred=lr.predict(x_test)

for i in range(len(pred)):
    if pred[i]>=0.5:
        pred[i]=1
    else:
        pred[i]=0
cm=confusion_matrix(y_test,pred)
print(cm)

#Backward elimination step 1
X=np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)
Xbw=X[:,[0,1,2,3,4,5]]
Xbw=np.array(Xbw,dtype=float)
model=sm.OLS(y,Xbw).fit()
print(model.summary())


#Backward elimination step 2
X=np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)
Xbw=X[:,[0,1,2,5]]
Xbw=np.array(Xbw,dtype=float)
model=sm.OLS(y,Xbw).fit()
print(model.summary())


