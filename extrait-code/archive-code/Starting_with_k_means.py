"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

import csv

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

_path = './extrait-code/artificial/'
_name="banana.arff"

def run_KMEANS_clustering(path, name, k, showplot):
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

    if showplot :
        #plt.figure(figsize=(6, 6))
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales : "+ str(name))
        #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
        plt.show()


    with open('results.csv', 'a', newline='') as results_file:
        # Write headers for the CSV files
        # Leave out but need to write beforehand
        # results_writer.writerow(['k', 'runtime_ms','inertia' ])
            
        # Run clustering method for a given number of clusters
        print("------------------------------------------------------")
        print("Appel KMeans pour une valeur de k fixée")
        tps1 = time.time()
        model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        # informations sur le clustering obtenu
        iteration = model.n_iter_
        inertie = model.inertia_
        centroids = model.cluster_centers_
        runtime = round((tps2 - tps1)*1000,2)

        # Entering data
        results_writer = csv.writer(results_file)
        results_writer.writerow([k, runtime, inertie])
        print( runtime, ", ", inertie)  

        if showplot :
            #plt.figure(figsize=(6, 6))
            plt.scatter(f0, f1, c=labels, s=8)
            plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
            plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
            #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
            plt.show()

            print("nb clusters =",k,", nb iter =",iteration, ", inertie = ", inertie, ", runtime = ", runtime,"ms")
            #print("labels", labels)

            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances(centroids)
            print(dists)


run_KMEANS_clustering(_path, _name, 2, 1)
