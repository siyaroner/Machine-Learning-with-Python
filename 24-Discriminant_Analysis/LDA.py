# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:44:44 2022

@author: Şiyar Öner
"""

#importing libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#importing csv file
data=pd.read_csv("wine.csv")
X=data.iloc[:,0:13].values
y=data.iloc[:,13].values

#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

#scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#LDA
lda=LinearDiscriminantAnalysis(n_components=2)
X_train2=lda.fit_transform(X_train,y_train)
X_test2=lda.transform(X_test)

lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

lr2=LogisticRegression(random_state=0)
lr2.fit(X_train2,y_train)
y_pred2=lr2.predict(X_test2)

cm=confusion_matrix(y_test, y_pred)
print("without lda((13 columns) cm:\n", cm)
cm2=confusion_matrix(y_test, y_pred2)
print("with lda(2 columns) cm2:\n", cm2)





