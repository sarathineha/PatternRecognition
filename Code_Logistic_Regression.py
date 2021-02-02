#importing all the needed libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import random
import heapq
import math

#loading iris dataset into iris variable
iris=load_iris()

#taking random 40 samples for train and 10 samples for test set from each class
n1=random.sample(range(0, 50), 40)
n1d=list(set(range(50))-set(n1))
n2=random.sample(range(50, 100), 40)
n2d=list(set(range(50,100))-set(n2))
n3=random.sample(range(100, 150), 40)
n3d=list(set(range(100,150))-set(n3))

#Generating Train And Test Set As Asked
X_train=np.zeros((120,4))
X_test=np.zeros((30,4))
y_train=np.zeros((120))
y_test=np.zeros((30))

c=0
for i in list(n1+n2+n3):
  X_train[c]+=iris.data[i]
  y_train[c]+=iris.target[i]
  c+=1
c=0
for i in list(n1d+n2d+n3d):
  X_test[c]+=iris.data[i]
  y_test[c]+=iris.target[i]
  c+=1



#Applying logistic Regression Function From Library to Classify

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=500)
model.fit(X_train,y_train)
y_pred_regression=model.predict(X_test)

#Printing predicted output of logistic regression
print(y_pred_regression)

from sklearn.metrics import accuracy_score

#Checking Accuracy Score for Logistic Regression
print(accuracy_score(y_test,y_pred_regression))