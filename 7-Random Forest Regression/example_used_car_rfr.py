# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:09:50 2022

@author: Şiyar Öner
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:33:32 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm
#importing data and analysing
data=pd.read_csv("UsedCarData.csv")

# print(data.dtypes)
df=data.drop(columns=["Sales_ID","name","Region","State or Province","City","seller_type","transmission","owner","torque","seats","sold"])
df=df[(df["fuel"]!="LPG")&(df["fuel"]!="CNG")]
# print(df.dtypes)
print(df.isnull().any())
print(df.describe())
print(df.corr().sort_values("selling_price"))
print(df.groupby("year").count().head(9))
df=df[df["year"]>2002]
# fig=plt.figure(figsize=(10,5))
# fig.set_dpi(500)
# plt.hist(df.iloc[:,0],label="year")
# # plt.hist(df.iloc[:,1],label="selling_price")
# # plt.hist(df.iloc[:,2],label="km driven")
# # plt.hist(df.iloc[:,4],label="mileage")
# # plt.hist(df.iloc[:,5],label="engine")
# # plt.hist(df.iloc[:,6],label="max power")
# plt.legend()
df=(pd.get_dummies(df)).drop(columns=["fuel_Petrol"])
print(df.columns)
y=df["selling_price"]
x=df.drop(columns=["selling_price"])

#test train split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


#rfr
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(x_train,y_train)
pred=rfr.predict(x_test)
r2=r2_score(y_test,pred)
mse=mean_squared_error(y_test, pred)

#backward elimination step 1
X=np.append(arr=np.ones((len(x_test),1)),values=x_test,axis=1)
X=X[:,[0,1,2,3,4,5,6]]
model=sm.OLS(y_test,X).fit()
print(model.summary())

#backward elimination step  2
X=np.append(arr=np.ones((len(x_test),1)),values=x_test,axis=1)
X=X[:,[0,1,2,3,5,6]]
model=sm.OLS(y_test,X).fit()
print(model.summary())

#backward elimination step 3
X=np.append(arr=np.ones((len(x_test),1)),values=x_test,axis=1)
X=X[:,[0,1,2,3,4,5]]
model=sm.OLS(y_test,X).fit()
print(model.summary())

#rfr 2
x_train2=x_train.drop(columns=["fuel_Diesel"])
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(x_train2,y_train)
X1=pd.DataFrame(X)
X1=X1.iloc[:,1:]
pred2=rfr.predict(X1)
r22=r2_score(y_test,pred2)
mse=mean_squared_error(y_test, pred2)
