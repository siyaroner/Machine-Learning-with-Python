# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:23:47 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#importing data and analysis
data=pd.read_csv("Social_Network_Ads.csv")
print(data.isnull().any())
print(data.describe())
print(data.corr())
x=data.drop(columns=["User ID","Purchased"])
x=pd.get_dummies(x)
x=x.drop(x.columns[-2],axis=1)
y=data["Purchased"]

# train test split and scaling
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)


#prediction Gini
dtc=DecisionTreeClassifier(criterion="gini")
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)

cm=confusion_matrix(y_test,pred)
print("gini")
print(cm)


#prediction Entropy
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)

cm=confusion_matrix(y_test,pred)
print("entropy")
print(cm)