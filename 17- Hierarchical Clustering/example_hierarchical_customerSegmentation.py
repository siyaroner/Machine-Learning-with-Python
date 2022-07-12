# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:49:45 2022

@author: Şiyar Öner
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
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
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()
# for n_clusters=5
hc=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
pred=hc.fit_predict(x,y)
plt.scatter(x["Income"],y,c=pred)
plt.show()
