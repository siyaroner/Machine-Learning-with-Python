# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:16:02 2022

@author: Şiyar Öner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cluster import KMeans

#import data
data=pd.read_csv("customerSegmentation.csv")
print(data.columns)
data=data.drop(columns=["Customer Id","Card Debt","Other Debt","Defaulted","Address"])
print(data.isnull().any())
print(data.describe())
print(data.corr())
sbn.pairplot(data)
plt.show()
data=data[data["Income"]<300]
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
y=np.array(y).reshape(-1,1)
#finding best n_clusters number
result=[]

for i in range(1,11):
    km=KMeans(n_clusters=i, init="k-means++",random_state=1)
    km.fit(x,y)
    result.append(km.inertia_)
    
plt.plot(range(1,11),result,"*-")
plt.show()
#pred
km=KMeans(n_clusters=5, init="k-means++",random_state=1)
pred=km.fit_predict(y)
pred=pd.DataFrame(pred)
# print(pd.unique(pred))
color=["green","blue","yellow","orange","red"]
pred=pred.replace([0,1,2,3,4],color)
print(pred.value_counts())
# color=["green","blue","yellow","orange","red"]
plt.scatter(x["Income"],y,c=pred)
