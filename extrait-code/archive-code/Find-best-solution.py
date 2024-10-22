import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from Starting_with_k_means import run_KMEANS_clustering
from Searching_for_best_k import search_k
from Starting_with_DBSCAN import run_DBSCAN_clustering
from Assessing_Neighbors import distanceToNeighbors
from Silhouette_optimal_k import optimal_k_silhouette

import csv
import pandas as pd
import math
import sys

def evaluate(algo, path, name):
    """
        Evaluates the best parameters tu run a given algorithm on a given data set
    """

    v = 5

    match algo:
        case "KMEANS":
            run_KMEANS_clustering(path, name, optimal_k_silhouette(path, name, 1), 1)

        case "AGLO":
            results = 0

        case "DBSCAN":   
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 1),  v, 1, 0)
            
        case "DBSCAN-STD":
            # Additional concern to asses the number v of neighbors taken into account
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 1), v, 0, 1)

evaluate(sys.argv[1],'./extrait-code/artificial/', sys.argv[2] )
