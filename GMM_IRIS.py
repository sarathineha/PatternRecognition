#importing libraries and loading dataset
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

iris = datasets.load_iris()
data = iris.data


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)

#Function to initialize the clusters

def initialize_clusters(X, n_clusters):
    clusters = []
    #rand.seed(1)
    
    center_index=rand.sample(range(len(X)),3)
    mu_k=[np.array(X[center_index[0]]),np.array(X[center_index[1]]),np.array(X[center_index[2]])]
    #taking any 3 random sample as centers/mean from data
    #center_index=rand.sample(range(len(X)),n_clusters)

    #creating mean array from taken samples for n clusters.
    #mu_k=[X[i] for i in range(n_clusters)]
    
    #creating dictionary which contains pi[k],mu[k]and co_var[k] for each kth-cluster
    
    for i in range(n_clusters):
        clusters.append({
            #setting pi 0.33 in starting as 1/3
            'pi_k': 1.0 / n_clusters,
            
            #setting mean
            'mu_k': mu_k[i],

            #setting identity matrix of covarience(here of 4x4)
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
        
    return clusters


#Function to calculate expectations
def expectation_step(X, clusters):

    #total variable to set gamma parameter
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)
    
    #calculating gamma value of each datapoint for each cluster
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        
        #gamma_nk contains gamma value of 150 data points for cluster =clustervalue of iteration
        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)
        
        #calculating total gamma value of each cluster for further use
        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]
        
        #setting new parameter in cluster dictionary
        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
        
    #dividing gamma of each datapoint to total of gamma value for that cluster as per formula
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']

#Function to minimize the value
def maximization_step(X, clusters):
    #taking N as dimension of data(here 4)
    N = float(X.shape[0])
  
    #updating each parameter of cluster
    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']

        #cov_k is dummy matrix containing zeros for further use
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        
        #calculating total gamma value of each cluster
        N_k = np.sum(gamma_nk, axis=0)
        
        #updating pi as per formula
        pi_k = N_k / N

        #updating mean values as per formula
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k
        
        #updating variance as per formula
        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            
        cov_k /= N_k
        
        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k

#Function to calculate log_likelyhood of clusters created
def get_likelihood(X, clusters):
    likelihood = []
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


#Function to itetate over dataset for GMM for given number of epochs
def gmm(X, n_clusters, n_epochs):
    #initializing clusters
    clusters = initialize_clusters(X, n_clusters)
    #initializing likelihood parameter variable
    likelihoods = np.zeros((n_epochs, ))
   
    #iterating n_epoch times
    for i in range(n_epochs):
        expectation_step(X, clusters)
        maximization_step(X, clusters)

        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood
        print('Iteration: ', i + 1, 'value of likelihood: ', likelihood)
    return clusters

#Applying GMM on IRIS
n_clusters = 3
n_epochs =300

clusters= gmm(data, n_clusters, n_epochs)



labels=[]
for i in range(150):
    temp=[clusters[0]['gamma_nk'][i],clusters[1]['gamma_nk'][i],clusters[2]['gamma_nk'][i]]
    labels.append(np.argmax(np.array(temp)))
print(labels)
print(adjusted_rand_score(iris.target,labels))