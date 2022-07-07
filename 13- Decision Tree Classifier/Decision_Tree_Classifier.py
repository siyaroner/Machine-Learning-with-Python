# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:34:53 2022

@author: Şiyar Öner
"""


#libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#importing data
data=pd.read_csv("data.csv")
print(data.columns)
x=data.iloc[:,1:4]
y=data["gender"]

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=6)

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

