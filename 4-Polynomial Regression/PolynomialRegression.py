# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:13:45 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import data
data=pd.read_csv("maaslar.csv")
x=data.iloc[:,1]
y=data.iloc[:,2]
x=np.array(x).reshape(-1,1)

#linear regression
lr1=LinearRegression()
lr1.fit(x,y)
plt.title("Degree 1")
plt.scatter(x,y,color="red")
plt.plot(x,lr1.predict(x))
plt.show()

#Polynomial Regression degree=2

pr1=PolynomialFeatures(2)
x_poly=pr1.fit_transform(x)
print(x_poly)
lr2=LinearRegression()
lr2.fit(x_poly,y)
plt.title("Degree 2")
plt.scatter(x,y,color="red")
plt.plot(x,lr2.predict(x_poly))
plt.show()


#Polynomial Regression degree=4

pr2=PolynomialFeatures(20)
x_poly=pr2.fit_transform(x)
print(x_poly)
lr3=LinearRegression()
lr3.fit(x_poly,y)
plt.title("Degree 20")
plt.scatter(x,y,color="red")
plt.plot(x,lr3.predict(x_poly))
plt.show()



print("linear degree 1")
print("6.5: ",lr1.predict([[6.5]]),"11:",lr1.predict([[11]]))
print("polynomial degree 2")
print("6.5: ",lr2.predict(pr1.fit_transform([[6.5]])),"10:",lr2.predict(pr1.fit_transform([[10]])))
print("polynomial degree 20")
print("6.5: ",lr3.predict(pr2.fit_transform([[6.5]])),"10:",lr3.predict(pr2.fit_transform([[10]])))









