from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

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

def optimal_agglo_silhouette(path, name, showplot):
    """
    Determines the optimal number of clusters (k) for Agglomerative clustering using the silhouette score.

    Args:
        path (str): The path to the directory containing the data file.
        name (str): The name of the data file (ARFF format).
        showplot (bool): If True, displays a plot of silhouette scores vs k values.

    Returns:
        int: The optimal k value with the highest silhouette score.
    """


    # Define the range of clusters k to test
    k_values = range(2, 50)  # You can adjust the range depending on your needs

    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])

    silhouette_scores = []
    
    for k in k_values:
        kmeans = AgglomerativeClustering(linkage='average', n_clusters=k)
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

optimal_agglo_silhouette('./extrait-code/artificial/', sys.argv[1], 1)