# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:53:12 2022

@author: Şiyar Öner
"""
# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

# reading data and slicing
df=pd.DataFrame(pd.read_csv("car data v2.csv"))
df=df[(df["selling_price"]<600000)&(df["km_driven"]<120000)]
# corrolation=df.corr()
df=df.drop(columns=["name","owner","seller_type","fuel","transmission"])
year=pd.DataFrame(df["year"])
mileage=pd.DataFrame(df["km_driven"])
price=pd.DataFrame(df["selling_price"])
milyear=pd.DataFrame(df[["year","km_driven"]])
print(mileage.count())

# plt.scatter(year,price)
print(price.mean())
# print(df.count())

# #split train test just for year
x_train,x_test,y_train,y_test=train_test_split(year,price,test_size=0.33, random_state=1)
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test.values,pred,"r")

#split train test just for mileage
# x_train,x_test,y_train,y_test=train_test_split(mileage,price,test_size=0.33, random_state=1)
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# pred=lr.predict(x_test)
# plt.scatter(x_test,y_test)
# plt.plot(x_test.values,pred,"r")

#split train test for mileage and year
# x_train,x_test,y_train,y_test=train_test_split(milyear,price,test_size=0.33, random_state=1)
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.fit_transform(x_test)
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# pred=lr.predict(x_test)
score=r2_score(y_test,pred)
err=mean_squared_error(y_test,pred)