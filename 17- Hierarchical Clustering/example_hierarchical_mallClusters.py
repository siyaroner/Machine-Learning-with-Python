# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:49:38 2022

@author: Şiyar Öner
"""


#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

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
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()
# for n_clusters=5
hc=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
pred=hc.fit_predict(x,y)
plt.scatter(x["Income"],y,c=pred)
plt.show()
plt.scatter(x["Income"],y,c=data["Clusters"])
plt.show()