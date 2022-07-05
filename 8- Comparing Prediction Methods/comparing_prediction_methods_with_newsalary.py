# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:45:56 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

#importing data and analysis
data=pd.read_csv("new_salaries.csv")
# print("null data?:\n",data.isnull().any())
# print("dtypes:\n",data.dtypes)
# print("corr:\n",data.corr())
df=data.drop(columns=["Calisan ID","unvan"])
x=df.iloc[:,0:3]
y=df["maas"]
# x=np.array(x).reshape(-1,3)
# sbn.pairplot(df,height=2.5)

######################### MLR ##################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=13)

# # Multiple Linear Regression
# mlr=LinearRegression()
# mlr.fit(x_train,y_train)
# pred_mlr=mlr.predict(x_test)
# r2_mlr=r2_score(y_test,pred_mlr)
# mse_mlr=mean_squared_error(y_test, pred_mlr)
# mae_mlr=mean_absolute_error(y_test, pred_mlr)


##################### PR ####################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=13)

# # fit predict
# pr=PolynomialFeatures(2)
# x_train_poly=pr.fit_transform(x_train)
# x_test_poly=pr.transform(x_test)
# plr=LinearRegression()
# plr.fit(x_train_poly,y_train)
# pred_plr=plr.predict(x_test_poly)
# r2_plr=r2_score(y_test,pred_plr)
# mse_plr=mean_squared_error(y_test,pred_plr)
# mae_plr=mean_absolute_error(y_test, pred_plr)



############## SVR #####################
# #train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=13)

# #Scaling
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.transform(x_test)
# y_train=np.array(y_train).reshape(-1,1)
# y_test=np.array(y_test).reshape(-1,1)
# y_train=sc.fit_transform(y_train)
# y_test=sc.transform(y_test)

# # fit predcit
# svr=SVR(kernel="rbf")
# svr.fit(x_train,y_train)
# pred_svr=svr.predict(x_test)
# pred=sc.inverse_transform(pred_svr.reshape(-1,1)).reshape(-1)
# r2_svr=r2_score(y_test,pred_svr)
# mse_svr=mean_squared_error(y_test, pred_svr)
# mae_svr=mean_absolute_error(y_test, pred_svr)


################### DTR ########################

# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# # Decision Tree
# dtr=DecisionTreeRegressor()
# dtr.fit(x_train,y_train)
# pred_dtr=dtr.predict(x_test)
# r2_dtr=r2_score(y_test,pred_dtr)
# mse_dtr=mean_squared_error(y_test, pred_dtr)
# mae_dtr=mean_absolute_error(y_test, pred_dtr)


############################### RFR ###############################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# # Random Forest
# rfr=RandomForestRegressor()
# rfr.fit(x_train,y_train)
# pred_rfr=rfr.predict(x_test)
# r2_rfr=r2_score(y_test,pred_rfr)
# mse_rfr=mean_squared_error(y_test, pred_rfr)
# mae_rfr=mean_absolute_error(y_test, pred_rfr)


# ## Backward Elimination
# # Step 1
# X=np.append(arr=np.ones((len(x),1)),values=x,axis=1)
# X=X[:,[0,1,2,3]]
# model=sm.OLS(y,X).fit()
# print(model.summary())


# # Step 2
# X=np.append(arr=np.ones((len(x),1)),values=x,axis=1)
# X=X[:,[0,1,2]]
# model=sm.OLS(y,X).fit()
# print(model.summary())


# # Step 3
# X=np.append(arr=np.ones((len(x),1)),values=x,axis=1)
# X=X[:,[0,1]]
# model=sm.OLS(y,X).fit()
# print(model.summary())

# # Step 1
# X=np.append(arr=np.ones((len(x),1)),values=x,axis=1)
# X=X[:,[1]]
# model=sm.OLS(y,X).fit()
# print(model.summary())

# AFTER BACKWARD ELIMINATION ###

x=df.iloc[:,0]
y=df["maas"]
x=np.array(x).reshape(-1,1)

######################### MLR ##################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=13)

# # Multiple Linear Regression
# mlr=LinearRegression()
# mlr.fit(x_train,y_train)
# pred_mlr=mlr.predict(x_test)
# r2_mlr=r2_score(y_test,pred_mlr)
# mse_mlr=mean_squared_error(y_test, pred_mlr)
# mae_mlr=mean_absolute_error(y_test, pred_mlr)


##################### PR ####################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=13)

# # fit predict
# pr=PolynomialFeatures(2)
# x_train_poly=pr.fit_transform(x_train)
# x_test_poly=pr.transform(x_test)
# plr=LinearRegression()
# plr.fit(x_train_poly,y_train)
# pred_plr=plr.predict(x_test_poly)
# r2_plr=r2_score(y_test,pred_plr)
# mse_plr=mean_squared_error(y_test,pred_plr)
# mae_plr=mean_absolute_error(y_test, pred_plr)



############## SVR #####################
# #train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=13)

# #Scaling
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.transform(x_test)
# y_train=np.array(y_train).reshape(-1,1)
# y_test=np.array(y_test).reshape(-1,1)
# y_train=sc.fit_transform(y_train)
# y_test=sc.transform(y_test)

# # fit predcit
# svr=SVR(kernel="rbf")
# svr.fit(x_train,y_train)
# pred_svr=svr.predict(x_test)
# pred=sc.inverse_transform(pred_svr.reshape(-1,1)).reshape(-1)
# r2_svr=r2_score(y_test,pred_svr)
# mse_svr=mean_squared_error(y_test, pred_svr)
# mae_svr=mean_absolute_error(y_test, pred_svr)


################### DTR ########################

# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# # Decision Tree
# dtr=DecisionTreeRegressor()
# dtr.fit(x_train,y_train)
# pred_dtr=dtr.predict(x_test)
# r2_dtr=r2_score(y_test,pred_dtr)
# mse_dtr=mean_squared_error(y_test, pred_dtr)
# mae_dtr=mean_absolute_error(y_test, pred_dtr)


############################### RFR ###############################
# # train test split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# # Random Forest
# rfr=RandomForestRegressor()
# rfr.fit(x_train,y_train)
# pred_rfr=rfr.predict(x_test)
# r2_rfr=r2_score(y_test,pred_rfr)
# mse_rfr=mean_squared_error(y_test, pred_rfr)
# mae_rfr=mean_absolute_error(y_test, pred_rfr)

