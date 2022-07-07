#libraries
import pandas as pd
import numpy as np
import seaborn as sbn

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#importing data and analysis
data=pd.read_csv("Social_Network_Ads.csv")
print(data.isnull().any())
print(data.describe())
print(data.corr())
x=data.drop(columns=["User ID","Purchased"])
x=pd.get_dummies(x)
x=x.drop(x.columns[-2],axis=1)
y=data["Purchased"]

# train test split and scaling
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fit and prediction
svc=SVC(kernel="rbf")
svc.fit(x_train,y_train)
pred=svc.predict(x_test)

cm=confusion_matrix(y_test,pred)
print(cm)


svc=SVC(kernel="sigmoid")
svc.fit(x_train,y_train)
pred=svc.predict(x_test)

cm=confusion_matrix(y_test,pred)
print(cm)
