# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:55:36 2022

@author: Şiyar Öner
"""
# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing,linear_model,metrics
from sklearn.model_selection import train_test_split

# carring missing values
data=pd.read_csv("missing_data.csv")
print(data)
imputer=SimpleImputer()
hwa=data.iloc[:,1:4].values #height weight age(hwa)
imputer = imputer.fit(hwa[:,1:4])
hwa[:,1:4] = imputer.transform(hwa[:,1:4])
print("hwa:\n",hwa)

#label and onehot encoder (turning categorical values into numeric values)
countries=data.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
countries[:,0]=le.fit_transform(data.iloc[:,0])
print("LabelEncoder \n",countries)

ohe=preprocessing.OneHotEncoder()
countries=ohe.fit_transform(countries).toarray()
print("onehotencoder \n",countries)
gender=data.iloc[:,-1].values
gender=le.fit_transform(gender)

#concating data
countriesdf=pd.DataFrame(data=countries,index=range(22),columns=["fr","tr","us"])
hwadf=pd.DataFrame(data=hwa,index=range(22),columns=["height","weight","age"])
genderdf=pd.DataFrame(data=gender,index=range(22),columns=["gender"])
df=pd.concat([countriesdf,hwadf,genderdf],axis=1)
print("concated df \n",df)

#data split
X=df.iloc[:,0:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)
scale=preprocessing.StandardScaler()
X_train.iloc[:,3:]=scale.fit_transform(X_train.iloc[:,3:])
X_test.iloc[:,3:]=scale.fit_transform(X_test.iloc[:,3:])
print(X_train)

#model selection and training
lr=linear_model.LinearRegression()

lr.fit(X_train,y_train)

pred=lr.predict(X_test)
for i in range(len(pred)):
    if pred[i]>=0.5:
        pred[i]=1 
    else: 
        pred[i]=0
pred=pd.DataFrame(pred)
cm=metrics.confusion_matrix(y_test,pred)
print(cm)










