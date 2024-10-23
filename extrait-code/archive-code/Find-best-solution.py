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

def evaluate(path, name, algo):
    """
    Executes and evaluates the specified clustering algorithm on the given dataset with the best parameters found.

    Args:
        algo (str): The clustering algorithm to use ("KMEANS", "DBSCAN", "DBSCAN-STD").
        path (str): The path to the directory containing the data file.
        name (str): The name of the data file (ARFF format).
    """

    v = 5

    match algo:
        case "KMEANS":
            run_KMEANS_clustering(path, name, optimal_k_silhouette(path, name, 0), 1)

        case "AGLO":
            results = 0

        case "DBSCAN":   
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 0),  v, 1, 0)
            
        case "DBSCAN-STD":
            # Additional concern to asses the number v of neighbors taken into account
            run_DBSCAN_clustering(path, name, distanceToNeighbors(path, name, v, 0), v, 0, 1)

evaluate('./extrait-code/artificial/', sys.argv[1], sys.argv[2] )
