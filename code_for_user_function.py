
#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

#Loading Dataset
df=load_iris()
X=df.data
target=df.target

"""# **Initializing Membership Matrix**"""

li=np.arange(150)
random.shuffle(li)
part1=li[:50]
part2=li[50:100]
part3=li[100:150]
for i in part1:
  membership_matrix[0][i]=1

for i in part2:
  membership_matrix[1][i]=1

for i in part3:
  membership_matrix[2][i]=1

"""# **Implementing Fuzzy C-means**"""

#setting m=2
m=2

#Creating membership matrix random values and centroid with zeros
membership_matrix=abs(np.array(membership_matrix))
centroid=np.zeros((3,4))

#Setting iteration=100
for it in range(100):

  #Code to generate centroid values using membership_matrix
  for i1 in range(3):
    temp=np.zeros((3,4))
    tot=0
    for j1 in range(150):
      temp[i1]+=(membership_matrix[i1][j1]**m)*X[j1]
      tot+=membership_matrix[i1][j1]**m
    centroid[i1]=temp[i1]/tot
  
  #Code to update membership_matrix values
  for i2 in range(150):
    tot=0
    for j2 in range(3):
      tot+=(1/np.linalg.norm(X[i2]-centroid[j2]))**(2/(m-1))
    print(tot)
    for k2 in range(3):
      membership_matrix[k2][i2]=((1/np.linalg.norm(X[i2]-centroid[k2])**(2/(m-1))))/tot
  
#getting clusters
prediction=np.argmax(membership_matrix,axis=0)
print(prediction)

"""# **Finding Match And Prining Accuracy**"""

#Code which creates a list of clusters predicted
lii=[]
li1=[i for i in range(len(predict)) if predict[i]==0]
li2=[i for i in range(len(predict)) if predict[i]==1]
li3=[i for i in range(len(predict)) if predict[i]==2]
lii.append(li1)
lii.append(li2)
lii.append(li3)

#Code which creates a list of clusters target
li=[]
l=[i for i in range(len(target)) if target[i]==0]
l1=[i for i in range(len(target)) if target[i]==1]
l2=[i for i in range(len(target)) if target[i]==2]
li.append(l)
li.append(l1)
li.append(l2)

#Implementing matching funtion to find accuracy
def do_match(li,li1):
  match=0
  for i in range(len(li)):
    maxx=0
    for j in range(len(li1)):
      if maxx<len(set(li[i]).intersection(set(li1[j]))):
        #print(len(set(li[i]).intersection(set(li1[j]))))
        maxx=len(set(li[i]).intersection(set(li1[j])))
    match+=maxx
  return match
print(do_match(li,lii))

print("The accuracy of fuzzey logic algorithm using algorithm is ",do_match(li,lii)/150)