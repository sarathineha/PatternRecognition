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






#Defining Eulidian Distance Function For KNN Algo
def euclid(data1,data2):
  temp=np.zeros(data1.shape)
  temp=(data1-data2)**2
  return math.sqrt(np.sum(temp))


#defining value of K as algo will consider K nearest samples
K=10

#Implementing KNN:- Calculating Euclidian distance of Each test samples from each Train samples and taking top 10 samples having lowest distance.
#From those 10 nearest samples, class for test sample is decided as the class which most of the samples-from selected 10 nearest samples belong to.
y_pred=[]
for i in range(len(X_test)):
  dis=[]
  for j in range(len(X_train)):
    dis.append(euclid(X_test[i],X_train[j]))
  n_smallest=heapq.nsmallest(K,range(len(np.array(dis))),np.array(dis).take)
  t1=np.zeros(3)
  for i in n_smallest:
    t1[int(y_train[i])]+=1
  y_pred.append(np.argmax(t1))

#Printing the predicted output
print(y_pred)

#Checking the accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

  


