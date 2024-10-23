import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import csv

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

from Starting_with_k_means import run_KMEANS_clustering

from kneed import DataGenerator, KneeLocator

import sys

_path = './extrait-code/artificial/'
_name="banana.arff"

def search_k(path, name, showplot):
    """
    Searches for the optimal number of clusters (k) for K-Means using the elbow method based on inertia.

    Args:
        path (str): The path to the directory containing the data file.
        name (str): The name of the data file (ARFF format).
        showplot (bool): If True, displays plots of inertia and runtime vs k values.

    Returns:
        int: The estimated optimal k value using the knee/elbow point detection.
    """

    with open('results.csv', 'w', newline='') as results_file :
        writer = csv.writer(results_file)

        writer.writerow(['k', 'runtime_ms', 'inertia'])
        results_file.close()

    # Define the range of clusters k to test
    k_values = range(2, 50)  # You can adjust the range depending on your needs
    

    for k in k_values:
        run_KMEANS_clustering(path, name, k, 0)

    data = pd.read_csv("results.csv")

    k_values = data['k']              # Number of clusters
    runtime = data['runtime_ms']    # Runtime in milliseconds
    inertia = data['inertia']         # Inertia

    if showplot:
        # Create the first plot (Inertia vs K)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertia, label='Inertia', marker='o', color='red')
        plt.title('Inertia vs Number of Clusters (K)')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Create the second plot (Runtime vs K)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, runtime, label='Runtime (ms)', marker='x', color='blue')
        plt.title('Runtime vs Number of Clusters (K)')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Runtime (ms)')
        plt.grid(True)
        plt.legend()
        plt.show()
    kneedle = KneeLocator(k_values, inertia, S = 1, curve='convex', direction='decreasing')  
    return kneedle.knee

search_k('./extrait-code/artificial/', sys.argv[1], 1)