import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from Assessing_Neighbors import distanceToNeighbors

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="xclara.arff"

# Nombre de voisins plus proches pour détermines epsilon
v = 5

def run_DBSCAN_clustering(path, name, showplot, standardized):

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

    # TODO : Watch out this is in a and none is in W, because not called by searching for best k
    with open('results.csv', 'a', newline='') as results_file:

        if not(standardized):
            # Run DBSCAN clustering method 
            # for a given number of parameters eps and min_samples
            # 
            print("------------------------------------------------------")
            print("Appel DBSCAN (1) ... ")
            tps1 = time.time()
            epsilon = distanceToNeighbors(path, name, v, 1)
            min_pts= v #5   # 10
            model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
            model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print("Chosen epsilon: ", epsilon)
            print("Minimum points: ", v)
            print('Number of clusters: %d' % n_clusters)
            print('Number of noise points: %d' % n_noise)

            if showplot :
                plt.scatter(f0, f1, c=labels, s=8)
                plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
                plt.show()
        else:
            ####################################################
            # Standardisation des donnees

            scaler = preprocessing.StandardScaler().fit(datanp)
            data_scaled = scaler.transform(datanp)
            print("Affichage données standardisées            ")
            f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
            f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

            if showplot:
                #plt.figure(figsize=(10, 10))
                plt.scatter(f0_scaled, f1_scaled, s=8)
                plt.title("Donnees standardisées")
                plt.show()


            print("------------------------------------------------------")
            print("Appel DBSCAN (2) sur données standardisees ... ")
            tps1 = time.time()
            epsilon= distanceToNeighbors(path, name, v, 1)
            min_pts= v # 10
            model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
            model.fit(data_scaled)

            tps2 = time.time()
            labels = model.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print('Number of clusters: %d' % n_clusters)
            print('Number of noise points: %d' % n_noise)

            if showplot:
                plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
                plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
                plt.show() 


#run_DBSCAN_clustering(path, name, 1, 0)
#run_DBSCAN_clustering(path, name, 1, 1)


