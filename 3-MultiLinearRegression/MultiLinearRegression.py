# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 15:16:20 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler

# import data & slicing
df=pd.read_csv("data.csv")
gender=df.iloc[:,-1]
countries=df["country"]
hwa=pd.DataFrame(df.iloc[:,1:-1])

#label and onehot encoding
le=LabelEncoder()
countries=le.fit_transform(countries)
y=le.fit_transform(gender)
ohe=OneHotEncoder()
countries=countries.reshape(-1,1)
countries=ohe.fit_transform(countries).toarray()

#scaling
sc=StandardScaler()
hwa=sc.fit_transform(hwa)

#concatenating data
hwa=pd.DataFrame(data=hwa,index=range(len(hwa)),columns=["height","weight","age"])
countries=pd.DataFrame(data=countries,index=range(len(hwa)),columns=["fr","tr","us"])
x=pd.concat([countries,hwa],axis=1)

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)

lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
for i in range(len(pred)):
    if pred[i]>=0.5:
        pred[i]=1
    else:
        pred[i]=0
print(pred)
cm=confusion_matrix(y_test,pred)
print(cm)
