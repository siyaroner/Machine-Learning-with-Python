# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:33:39 2022

@author: Şiyar Öner
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger 
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
#generatin non-linear data sets
X,y=make_circles(100,factor=.1,noise=.1)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="bwr")

#rain test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# fit and prediction
svc=SVC(kernel="linear")
svc.fit(X_train,y_train.ravel())
pred=svc.predict(X_test)

acc=accuracy_score(y_test,pred)
print(acc)


zero_one_colormap=ListedColormap(("blue","red"))

def plot_decision_boundary(X,y,clf):
    X_set,y_set =X,y
    X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                                stop=X_set[:,0].max()+1,
                                step=0.01),
                      np.arange(start=X_set[:,1].min()-1,
                                stop=X_set[:,1].max()+1,
                                step=0.01))
    plt.contourf(X1,X2,clf.predict(np.array([X1.ravel(),
                                               X2.ravel()]).T).reshape(X1.shape),
                   alpha=0.75,
                   cmap=zero_one_colormap)
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                    c=(zero_one_colormap)(i),label=j)
        plt.title("svm decision boundary")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        return plt.show()
    
# plot_decision_boundary(X, y, svc2)
    
# def plot_3d_plot(X,y):
#     r=np.exp(-(X**2).sum(1))
#     ax=plt.subplot(projection="3d")
#     ax.scatter3D(X[:,0],X[:,1],r,c=y,s=100,cmap="bwr")
#     ax.set_xlabel("X1")
#     ax.set_ylabel("X2")
#     ax.set_zlabel("y")
#     return ax
# plot_3d_plot(X, y)

# svc2=SVC(kernel="rbf")
# svc2.fit(X_train,y_train)
# pred2=svc2.predict(X_test)
# acc2=accuracy_score(y_test,pred2)
# print(acc2)
# plot_decision_boundary(X, y, svc2)


svc3 = SVC(kernel="poly",degree=2)
svc3.fit(X_train, y_train)
pred3 = svc3.predict(X_test)

acc3=accuracy_score(y_test, pred3)
print(acc3)
plot_decision_boundary(X, y, svc3)










