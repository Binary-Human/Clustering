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
import sys

def evaluate_runtime_DBSCAN(path, name):
    """
        Evaluates runtime of DBSCAN for various values of epsilon on given dataset
    """

    with open('results.csv', 'w', newline='') as results_file :
        writer = csv.writer(results_file)

        writer.writerow(['e', 'runtime_ms', 'inertia'])
        results_file.close()

    epsilon = distanceToNeighbors(path, name, 5, 0)
    step = epsilon/10
    e = 0 

    while (e<epsilon*2.0):
        e += step 
        run_DBSCAN_clustering(path, name, e, 5, 0, 0)

    data = pd.read_csv("results.csv")

    k_values = data['e']              # Number of clusters
    runtime = data['runtime_ms']    # Runtime in milliseconds

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, runtime, label='Runtime (ms)', marker='x', color='blue')
    plt.title('Runtime vs Epsilon')
    plt.xlabel('Epsilon (e)')
    plt.ylabel('Runtime (ms)')
    plt.grid(True)
    plt.legend()
    plt.show()

evaluate_runtime_DBSCAN('./extrait-code/artificial/', sys.argv[1] )