# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:11:03 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cluster import KMeans

#import data
data=pd.read_csv("mallClusters.csv")
print(data.describe())
plt.figure(figsize=(15,15),dpi=500)
sbn.pairplot(data[["Age","Income","Spending_Score"]])
plt.show()
data=pd.get_dummies(data)
x=data[["Age","Income","Gender_Male"]]
y=data["Spending_Score"]
# x=np.array(x).reshape(-1,1)
#finding best n_clusters number
result=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init="k-means++",random_state=1)
    km.fit_predict(x)
    result.append(km.inertia_)

plt.plot(range(1,11),result,"*-")
plt.show()
# for n_clusters=5
km=km=KMeans(n_clusters=5,init="k-means++",random_state=100)
pred=km.fit_predict(x,y)
cluster_centers=np.array(km.cluster_centers_)
print(np.array(cluster_centers))
plt.scatter(x["Income"],y,c=data["Clusters"])
plt.show()
plt.scatter(x["Income"],y,c=pred)

