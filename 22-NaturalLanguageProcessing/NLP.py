# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:03:38 2022

@author: Şiyar Öner
"""

#libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem.porter import PorterStemmer as ps
nltk.download("stopwords")
from nltk.corpus import stopwords

#data cleaning make it appropriate for csv file
# reviews=pd.read_excel("Restaurant_Reviews.xlsx",header=None)
# for i in range(len(reviews)):
#         review=reviews.iloc[i,0]
#         review=re.sub("[^a-zA-Z-0-9]"," ",review)
#         review=list(review)
#         for j in range(len(review)):
#             if len(review)==j+1 and (ord(review[j])==48 or ord(review[j])==49) :
#                 review.insert(j,",")
#         
          ##way 1
          # review=review.join 
          ##way 2
          # str1=""
#         for k in review:
#             str1+=k.lower()
#         reviews.iloc[i,0]  =str1
# reviews.iloc[0,0]="Review,Liked"
# # print(reviews.iloc[0,0])   
# reviews.to_csv('reviews.csv',index=False)  #finally open csv file and delete first row which just 0

#data cleaning is done now let's import it
df=pd.read_csv("reviews.csv")


# Reading dataset row by row 
filtered_words=[]

for i in range(len(df)):
    review=df["Review"][i]
    review=review.split()
    review=[ps().stem(word) for word in review if not word in set(stopwords.words('english'))] # stem is using to find root of the word
    review=" ".join(review)
    filtered_words.append(review)
# data preprocessing
cv=CountVectorizer (max_features=2000)
x=cv.fit_transform(filtered_words).toarray()
y=df["Liked"].values

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=2)

#model selection
ghb=GaussianNB()

ghb.fit(x_train,y_train)
y_pred=ghb.predict(x_test)

#test
cm=confusion_matrix(y_test,y_pred)
print(cm)















