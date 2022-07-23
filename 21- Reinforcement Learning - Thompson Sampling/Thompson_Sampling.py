# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:17:35 2022

@author: Siyar Oner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

data=pd.read_csv("Ads_CTR_Optimisation.csv")

N = len(data)
ads = len(data.columns)
chosens=[]
total_rewards_per_ad = [1]*ads
total_reward=0
ones=[1.1]*ads
zeros=[1.1]*ads
for i in range(0,N): 
    selected_ad = total_rewards_per_ad.index(max(total_rewards_per_ad))
    max_th = 0
    for j in range(0, ads):
        # for initialize
        rasbeta=random.betavariate(ones[j],zeros[j])
        
        if  max_th<=rasbeta:
            max_th = rasbeta
            selected_ad = j
            if i%50==0 & selected_ad!=total_rewards_per_ad.index(max(total_rewards_per_ad)):
                selected_ad = total_rewards_per_ad.index(max(total_rewards_per_ad))
    chosens.append(selected_ad)            
    reward = data.values[i, selected_ad]
    total_rewards_per_ad[selected_ad] += reward
    total_reward+=reward
    if reward==1:
        ones[selected_ad]+=1
    else:
        zeros[selected_ad]+=1

#total_reward = sum(total_rewards_per_ad)
print(total_reward)
plt.hist(chosens)
plt.show()
