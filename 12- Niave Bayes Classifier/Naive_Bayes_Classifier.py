# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:34:53 2022

@author: Şiyar Öner
"""


#libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix

#importing data
data=pd.read_csv("data.csv")
print(data.columns)
x=data.iloc[:,1:4]
y=data["gender"]

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=6)

#prediction GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
pred=gnb.predict(x_test)

cm=confusion_matrix(y_test,pred)
print("gnb")
print(cm)


#prediction MultinomialNB
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
pred=mnb.predict(x_test)

cm=confusion_matrix(y_test,pred)
print("mnb")
print(cm)

#prediction BernoulliNB
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
pred=bnb.predict(x_test)

cm=confusion_matrix(y_test,pred)
print("bnb")
print(cm)