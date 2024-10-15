import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from Starting_with_DBSCAN import run_DBSCAN_clustering

def evaluate(k_max, algo, path, name):

    results = []
    inertia = []
    runtime = []
    n_clusters = []
    n_outliers = []

    match algo:
        case "KMEANS":
            for i in range(k_max):
                results = 0

        case "AGLO":
            for i in range(k_max):
                results = 0

        case "DBSCAN":
            for i in range(k_max):
                # Additional concern to asses the number v of neighbors taken into account
                run_DBSCAN_clustering(path, name, 0, 0)
                # TODO get the equivalent of searching for best k, with number of clusters i guess maybe
                # Could be an if with the number of clusters
                # Or an assement of inertia, elbow method
        
        case "DBSCAN-Std":
            for i in range(k_max):
                # Additional concern to asses the number v of neighbors taken into account
                run_DBSCAN_clustering(path, name, 0, 1)

                


# Run les algorithmes avec plusieurs paramètres
# Trouver une métrique d'évaluation du meilleur run
# Refactor models into functions
# Récupere les données du runtime et inertie

# Comment faire quand ca n'est pas straightforward ?
