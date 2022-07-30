# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:39:32 2022

@author: Şiyar Öner
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

#import data and analysis
data=pd.read_csv("ChurnModelling.csv")

# #to see data types of the data
# print(data.dtypes)

# #we don't need Rownumber,CostermerId,Surname columns for our model so we won't use them
# # And we have two columns which contain categorical values so we gotta encode them with
# # label and one hot encoder
X=data.iloc[:,3:13]
y=data.iloc[:,13]
le=LabelEncoder()
X["Gender"]=le.fit_transform(X["Gender"])
le2=LabelEncoder()
countries=le2.fit_transform(X["Geography"])
ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])], remainder="passthrough")
X=ohe.fit_transform(X)
X=X[:,1:]

# #Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
y_pred=xgbc.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))