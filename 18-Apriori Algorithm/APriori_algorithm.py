# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:59:15 2022

@author: Şiyar Öner
"""

import pandas as pd
import numpy as np

from apyori import apriori

data=pd.read_csv("sepet.csv",header=None)
print(len(data))
lst=[]
for i in range (0,len(data)):
    lst.append([str(data.values[i,j])for j in range(0,20)])
    
print(list(apriori(lst,min_support=0.008,min_confidence=0.2,min_lift=3)))

