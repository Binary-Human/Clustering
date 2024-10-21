import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from Starting_with_DBSCAN import run_DBSCAN_clustering
from Assessing_Neighbors import distanceToNeighbors

import csv
import pandas as pd
import math

def evaluate(k_max, algo, path, name):

    results = []
    inertia = []
    runtime = []
    n_clusters = []
    n_outliers = []

    v = 5

    match algo:
        case "KMEANS":
            for i in range(k_max):
                results = 0

        case "AGLO":
            for i in range(k_max):
                results = 0

        case "DBSCAN":   
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 1),  v, 1, 0)
            
        case "DBSCAN-STD":
            # Additional concern to asses the number v of neighbors taken into account
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 1), v, 0, 1)

                


# Run les algorithmes avec plusieurs paramètres
# Trouver une métrique d'évaluation du meilleur run
# Refactor models into functions
# Récupere les données du runtime et inertie

# Comment faire quand ca n'est pas straightforward ?
evaluate(10, "DBSCAN",'./extrait-code/artificial/', "xclara.arff" )