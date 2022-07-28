# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:39:32 2022

@author: Şiyar Öner
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
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
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# # Artificial Neural Network

classifier=Sequential()
classifier.add(Dense(6,activation="relu",input_dim=11)) #init for w, uniform for a value which close to 0
classifier.add(Dense(6, activation="relu"))
classifier.add(Dense(1,activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,epochs=200)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>=0.5)
cm=confusion_matrix(y_test,y_pred)
print(cm)
