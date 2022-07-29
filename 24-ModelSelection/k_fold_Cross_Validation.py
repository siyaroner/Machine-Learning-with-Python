# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:37:16 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import seaborn as sbn

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#importing data and analysis
data=pd.read_csv("Social_Network_Ads.csv")
# print(data.isnull().any())
# print(data.describe())
# print(data.corr())
x=data.drop(columns=["User ID","Purchased"])
x=pd.get_dummies(x)
x=x.drop(x.columns[-2],axis=1)
y=data["Purchased"]

# train test split and scaling
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fit and prediction
svc=SVC(kernel="rbf")
svc.fit(x_train,y_train)
pred=svc.predict(x_test)

#Confusion Matrix
cm=confusion_matrix(y_test,pred)
print(cm)

#k-fold Cross Validation Score
"""
1- estimator=svc
2- X
3- y
+- cv= how many folds will it be
"""
val_score=cross_val_score(estimator=svc, X=x_train,y=y_train,cv=4)
print(val_score)
print("mean val_score:",val_score.mean())



















