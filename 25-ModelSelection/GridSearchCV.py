# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:34:53 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import seaborn as sbn

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
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

# #k-fold Cross Validation Score
# """
# 1- estimator=svc
# 2- X
# 3- y
# +- cv= how many folds will it be
# """
# val_score=cross_val_score(estimator=svc, X=x_train,y=y_train,cv=4)
# print(val_score)
# print("mean val_score:",val_score.mean())

##Grid Search
# optimazing parameters and selecting algorithm

p=[{'C':[1,2,3,4,5],'kernel':['linear','rbf']},
   {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]}]

"""
GSCV parameters:
    1-estimator: classification algorithm (which we wanna optimize)
    2- pram-grid: parameters/the values we'll try
    3-scoring: with which method it'll evaluate
    4- cv: how many folds will it be
    5- n_jobs: the number of works which will execute at the same time
    
"""
gs=GridSearchCV(estimator=svc, param_grid=p,scoring="accuracy",cv=10,n_jobs=-1)

grid_search=gs.fit(x_train,y_train)
theBestScore=grid_search.best_score_
theBestParameters=grid_search.best_params_

print("the best score:",theBestScore,"\n the best parameters:",theBestParameters)














