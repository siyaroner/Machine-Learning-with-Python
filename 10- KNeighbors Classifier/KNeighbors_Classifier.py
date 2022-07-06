# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:05:49 2022

@author: Şiyar Öner
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#importing data
data=pd.read_csv("data.csv")
print(data.columns)
x=data.iloc[:,1:4]
y=data["gender"]

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#fit predict
knc=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knc.fit(x_train,y_train)
pred=knc.predict(x_test)
cm=confusion_matrix(y_test,pred)
print(cm)