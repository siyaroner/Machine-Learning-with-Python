# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 18:57:28 2022

@author: Şiyar Öner
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,roc_auc_score


#importing data and analysis
data=pd.read_excel("iris.xls")
# print(data.isnull().any())
# print(data.describe())
# print(data.corr())
df=pd.get_dummies(data)
x=data.iloc[:,0:4]
y=data.iloc[:,4:]


# # train test split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)

# RFC
rfc=RandomForestClassifier(n_estimators=150, criterion="entropy")
rfc.fit(x_train,y_train)
pred=rfc.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("RFC")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)

# DTC
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("DTC")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)

# #GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
pred=gnb.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("GaussianNB")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)

# #KNC
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
pred=knc.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("KNC")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)

# #Scalling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# #SVC
svc=SVC(kernel="rbf")
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("SVC")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)

# #LR
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
cm=confusion_matrix(y_test, pred)
print("LR")
print(cm)
# Roc_Auc_Score=roc_auc_score(y_test,pred)
# print(Roc_Auc_Score)