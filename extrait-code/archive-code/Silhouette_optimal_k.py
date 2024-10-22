from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

import csv
import pandas as pd
import math
import sys

def optimal_k_silhouette(path, name, showplot):
    """
        Evaluates best k value for given dataset, by analysing silhouette index
        - Showplot : Boolean to show the graph corresponding the evolution of inertia and runtime
    """


    # Define the range of clusters k to test
    k_values = range(2, 50)  # You can adjust the range depending on your needs

    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])

    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(datanp)
        if k > 1:
            score = silhouette_score(datanp, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # Silhouette score is not defined for k=1
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
        
    if showplot:
        plt.plot(k_values, silhouette_scores, 'bo-')
        plt.title(f'Optimal number of clusters: {optimal_k} (Silhouette Score)')
        plt.show()
        
    return optimal_k

optimal_k_silhouette('./extrait-code/artificial/', sys.argv[1], 1)