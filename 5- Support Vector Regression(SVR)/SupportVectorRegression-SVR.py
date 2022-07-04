# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:44:23 2022

@author: Şiyar Öner
"""
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


#Importing Data and slicing
data=pd.read_csv("maaslar.csv")
x=data.iloc[:,1]
y=data.iloc[:,2]
x=np.array(x).reshape(-1,1)

#linear regression
lr=LinearRegression()
lr.fit(x,y)
plt.title("linear regression")
plt.scatter(x,y,color="red")
plt.plot(x,lr.predict(x))
plt.show()

# polynomial regression degree 2
pl=PolynomialFeatures(2)
x_poly=pl.fit_transform(x)
lr2=LinearRegression()
lr2.fit(x_poly,y)
plt.title("polynomial regression degree 2")
plt.scatter(x,y,color="red")
plt.plot(x,lr2.predict(x_poly))
plt.show()

#polynomial regression degree 4
pl2=PolynomialFeatures(4)
x_poly2=pl2.fit_transform(x)
lr3=LinearRegression()
lr3.fit(x_poly2,y)
plt.title("polynomial regression degree 4")
plt.scatter(x,y,color="red")
plt.plot(x,lr3.predict(x_poly2))
plt.show()
print("linear")
print("11:", lr.predict([[11]]),"6.5",lr.predict([[6.5]]))
print("polynomial degree 2")
print("11:", lr2.predict(pl.fit_transform([[11]])),"6.5",lr2.predict(pl.fit_transform([[6.5]])))
print("polynomial degree 4")
print("11:", lr3.predict(pl2.fit_transform([[11]])),"6.5",lr3.predict(pl2.fit_transform([[6.5]])))

#scalling data
sc=StandardScaler()
x=sc.fit_transform(x)
y=np.array(y).reshape(-1,1)
y=sc.fit_transform(y)
#support vector regression rbf
svr=SVR(kernel="rbf")
svr.fit(x,y)
plt.title("svr")
fig = plt.figure(figsize =(5, 4)) 
fig.set_dpi(500)
plt.scatter(x,y,color="red")
plt.plot(x,svr.predict(x),label="rbf")
pred1=svr.predict([[11]])
pred2=svr.predict([[6]])
pred1 = sc.inverse_transform(pred1.reshape(-1,1)).reshape(-1)
pred2=sc.inverse_transform(pred2.reshape(-1,1)).reshape(-1)
print( "svr with rbf")
print("11:",pred1,"6.5",pred2)

#support vector regression polynomial
svr=SVR(kernel="poly", degree=6, coef0=1)
svr.fit(x,y)
plt.title("svr")
plt.scatter(x,y,color="red")
plt.plot(x,svr.predict(x),label="poly")
pred1=svr.predict([[11]])
pred2=svr.predict([[6]])
pred1 = sc.inverse_transform(pred1.reshape(-1,1)).reshape(-1)
pred2=sc.inverse_transform(pred2.reshape(-1,1)).reshape(-1)
print( "svr with poly")
print("11:",pred1,"6.5",pred2)

# #support vector regression linear
svr=SVR(kernel="linear")
svr.fit(x,y)
plt.title("svr")
plt.scatter(x,y,color="red")
plt.plot(x,svr.predict(x),label="Linear")
pred1=svr.predict([[11]])
pred2=svr.predict([[6]])
pred1 = sc.inverse_transform(pred1.reshape(-1,1)).reshape(-1)
pred2=sc.inverse_transform(pred2.reshape(-1,1)).reshape(-1)
print( "svr with linear")
print("11:",pred1,"6.5",pred2)


#support vector regression sigmoid
svr=SVR(kernel="sigmoid")
svr.fit(x,y)
plt.title("svr")
plt.scatter(x,y,color="red")
plt.plot(x,svr.predict(x),label="Sigmoid")
pred1=svr.predict([[11]])
pred2=svr.predict([[6]])
pred1 = sc.inverse_transform(pred1.reshape(-1,1)).reshape(-1)
pred2=sc.inverse_transform(pred2.reshape(-1,1)).reshape(-1)
print( "svr with linear")
print("11:",pred1,"6.5",pred2)

plt.legend()

