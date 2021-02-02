
#code 1.1
#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
import pandas as pd
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df = sklearn_to_df(load_iris())

df.head()




#code 1.2
#taking all 4 features of dataset into a dataframe X
X = df[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]]

#Defining number of clusters
K=3

# Selecting centroids by taking random K rows from dataset X
Centroids = (X.sample(n=K))

print("Printing the Difference of Centroids")
diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    #iterating over all rows in X and finding euclidian distance between each point's feature value to its corresponding centroid value
    for index1,row_c in Centroids.iterrows():
        #a list which shows euclidian distance of a point to all the centroids available , it is in list of list format
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["sepal length (cm)"]-row_d["sepal length (cm)"])**2
            d2=(row_c["sepal width (cm)"]-row_d["sepal width (cm)"])**2
            d3=(row_c["petal length (cm)"]-row_d["petal length (cm)"])**2
            d4=(row_c["petal width (cm)"]-row_d["petal width (cm)"])**2
            d=np.sqrt(d1+d2+d3+d4)
            ED.append(d)
        X[i]=ED
        i=i+1

    #Code to find nearest centroid to each point
    #list C stores position or index of centroid nearest to a point
    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
      
    #assigning list C to dataset X as "cluster" column
    X["Cluster"]=C
    #Finding mean value for all features for created clusters 1,2,and 3
    Centroids_new = X.groupby(["Cluster"]).mean()[["sepal width (cm)","sepal length (cm)","petal length (cm)","petal width (cm)"]]
    #If our iteration is first one when the centroids are randomly taken,then ignoring that centroid values and going to another iteration so that both the centroid values(new/old) are be estimated
    if j == 0:
        diff=1
        j=j+1
    else:
        #finding difference by taking sum of diference between all the features and taking addition of all those sum to get overall difference
        diff = (Centroids_new['sepal width (cm)'] - Centroids['sepal width (cm)']).sum() + (Centroids_new['sepal length (cm)'] - Centroids['sepal length (cm)']).sum()+(Centroids_new['petal length (cm)'] - Centroids['petal length (cm)']).sum()+(Centroids_new['petal width (cm)'] - Centroids['petal width (cm)']).sum()
        print(diff.sum())
    #assigning current new_centroids to Centroids to use it in next iteration
    Centroids = X.groupby(["Cluster"]).mean()[["sepal width (cm)","sepal length (cm)","petal length (cm)","petal width (cm)"]]
print()
print()





#code 1.3
#creating and printing all clusters created
#appending all clusters to a list li
li=[]
y=X.groupby(["Cluster"]).groups
print("Clusters Created Using Algorithm:")
for i in y:
  li.append(y[i].tolist())
  print(y[i])
print()
print()







#code 2.1
#doing clustering using kmeans function of sklearn API 

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets#Iris Dataset
iris1 = datasets.load_iris()
X1 = iris1.data#KMeans
km = KMeans(n_clusters=3)
km.fit(X1)
km.predict(X1)
labels = km.labels_

#creating a column in dataframe X with classified values of each row using kmeans-API
X["clusterwithAPI"]=labels





#code 2.2
#creating a list which contains all clusters generated using kmeans function of API
#printing the list
li1=[]
y1=X.groupby(["clusterwithAPI"]).groups
print("Clusters Created using API")
for i in y1:
  li1.append(y1[i].tolist())
  print(y1[i])
print()
print()




#code 3.1
#creating a list of target values of all rows from dataset for checking accuracy
li2=[]
y2=df.groupby(["target"]).groups
print("Actual Targer Classes of IRIS")
for i in y2:
  li2.append(y2[i].tolist())
  print(y2[i])
print()
print()




#code 3.2
#function to match rows of clusters using kmeans algorithm and the values to actual dataset
#Note-Kmeans clustering generated only clusters, so any label can contain any cluster.
#to match the clustering , i have taken maximum intersection between clusters
#if cluster labeled "0" has maximum intersection to target class "3" then, code will map label-"0" to target class "3" and will take intersection
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





#code 3.3
#Code which prints total exact match of classfication 
#It also prints accuracy score

l=[y2[i].tolist() for i in y2]
method_y=[y[i].tolist() for i in y]
api_y=[y1[i].tolist() for i in y1]
match1=do_match(l,method_y)
match2=do_match(l,api_y)
print("The found total match between Kmeans Algorithm and Actual Dataset values are",match1)
print("The accuracy of implemented Kmeans algorithm is",match1/150)
print("The found total match between Kmeans API Algorithm and Actual Dataset values are",match2)
print("The accuracy of API-Kmeans algorithm is",match2/150)