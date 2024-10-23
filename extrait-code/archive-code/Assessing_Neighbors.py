import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import csv
import math
import sys

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
 
def distanceToNeighbors(path, name, v, showplot) :
    """
    Calculates the distance to the v-th nearest neighbor for each data point
    and returns the 98th percentile of these distances. This can be used to 
    estimate a suitable epsilon value for DBSCAN.

    Args:
        path (str): The path to the directory containing the data file.
        name (str): The name of the data file (ARFF format).
        v (int): The number of nearest neighbors to consider.
        showplot (bool): If True, displays a plot of the sorted average distances.

    Returns:
        float: The 98th percentile of the average distances to the v nearest neighbors.
    """
    
    #path_out = './fig/'
    databrut = arff.loadarff(open(path+str(name), 'r'))
    datanp = np.array([[x[0],x[1]] for x in databrut[0]])

    # PLOT datanp (en 2D) - / scatter plot
    # Extraire chaque valeur de features pour en faire une liste
    # EX : 
    # - pour t1=t[:,0] --> [1, 3, 5, 7]
    # - pour t2=t[:,1] --> [2, 4, 6, 8]
    print("---------------------------------------")
    print("Affichage données initiales            "+ str(name))
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne

    # Distances aux k plus proches voisins
    # Donnees dans X
    v = 5
    neigh = NearestNeighbors(n_neighbors = v)
    neigh.fit(datanp)
    distances , indices = neigh.kneighbors(datanp)

    # distance between points
    #
    # distance moyenne sur les k plus proches voisins
    # en retirant le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])] )
    # trier par ordre croissant
    distancetrie = np.sort(newDistances) 

    if showplot :
        plt.title(f"Distances aux {str(v)} proches voisins pour tous les points" )
        plt.xlabel('"id" du point dans le modèle')
        plt.ylabel("distance")
        plt.plot(distancetrie)
        plt.show()

    return np.percentile(distancetrie,98)

distanceToNeighbors('./extrait-code/artificial/', sys.argv[1], sys.argv[2], 1)
