# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:42:23 2022

@author: Şiyar Öner
"""
##libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle
from pypmml import Model
url="https://bilkav.com/satislar.csv"
data=pd.read_csv(url)
X=data.iloc[:,0:1]
Y=data.iloc[:,1]
 
test_size=0.33

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,random_state=1)

lr=LinearRegression()
lr.fit(X_train,y_train)

file="model.save"
pickle.dump(lr,open(file,"wb")) #write binary

loaded=pickle.load(open(file,"rb"))
y_pred=loaded.predict(X_test)
print(y_pred)


## other methods: pmml and joblib
