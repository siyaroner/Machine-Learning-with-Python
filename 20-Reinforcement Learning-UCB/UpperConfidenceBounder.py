# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:57:15 2022

@author: Şiyar Öner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv("Ads_CTR_Optimisation.csv")

N = len(data)
ads = len(data.columns)
chosens = []
selection_numbers_per_ad = [0]*ads
total_rewards_per_ad = [0]*ads
max_ucb = 0
for i in range(0,N):
    
    selected_ad = total_rewards_per_ad.index(max(total_rewards_per_ad))
    for j in range(0, ads):
        # for initialize
        if selection_numbers_per_ad[j] > 0:
            mean=total_rewards_per_ad[j]/selection_numbers_per_ad[j]
            delta_i=math.sqrt(3/2*math.log(i)/selection_numbers_per_ad[j])
            ucb=mean+delta_i
        else:
            ucb = 1e1000
        if ucb > max_ucb:
            max_ucb = ucb
            selected_ad = j
    chosens.append(selected_ad)
    selection_numbers_per_ad[selected_ad] += 1
    reward = data.values[i, selected_ad]
    total_rewards_per_ad[selected_ad] += reward

total_reward = sum(total_rewards_per_ad)
print(total_reward)
