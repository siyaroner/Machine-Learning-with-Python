# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:58:42 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#importing data
data=pd.read_csv("data.csv")
print(data.columns)
x=data.iloc[:,1:4]
y=data["gender"]

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#prediction
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)




