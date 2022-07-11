# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:42:16 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#importing data
data=pd.read_csv("customers.csv")
y=data["volume"]
x=data["salary"]
x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)

#kmeans fit
km=KMeans(n_clusters=3,init="k-means++")
pred=km.fit_predict(x,y)
plt.figure(figsize=(12,12),dpi=500)
plt.subplot(221)
plt.scatter(x,y,c=pred)
plt.title("n_clusters=3")
plt.xlabel("salary")
plt.ylabel("volume")
plt.show()
print(km.cluster_centers_)

#finding best n_cluster by elbow method
result=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,init="k-means++",random_state=1)
    km.fit(x)
    result.append(km.inertia_) #inertia: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    
plt.plot(range(1,10),result,"*-")


#4 is best n_cluster
km=KMeans(n_clusters=4,init="k-means++",random_state=1)
pred=km.fit_predict(x,y)
cluster_centers=km.cluster_centers_
print(cluster_centers)
plt.figure(figsize=(12,12),dpi=500)
plt.subplot(222)
plt.scatter(x,y,c=pred)
plt.title("n_clusters=4")
plt.xlabel("salary")
plt.ylabel("volume")
# plt.scatter(
#     cluster_centers[0, :],
#     cluster_centers[1, :],
#     marker="x",
#     s=169,
#     linewidths=3,
#     color="r",
#     zorder=10,
# )
plt.show()