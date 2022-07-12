# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:02:48 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
#importing data
data=pd.read_csv("customers.csv")
x=data.iloc[:,3:]

#Hierarchical Clustering/Agglomerative
ac=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
pred=ac.fit_predict(x)
print(pred)
plt.scatter(x["volume"],x["salary"],s=30,c=pred)
plt.show()

dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()

