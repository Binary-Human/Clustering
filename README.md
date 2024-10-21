# Clustering
5SDBD - Apprentissage non-supervisé

## Prerequisites

## Libraries

### Before running
Need to go to extrait code to work

## Running the code

### Different options

Most of what happends is through the find-best-solution script

OPTIONS :
    - "KMEANS" : Runs k-means through 
    - "AGLO" : 
    - "DBSCAN" : Runs DBSCAN with the most adequate parameters for epsilon, thanks to distanceToNeighbors in Assessing_Neighbors
    ```python
        def distanceToNeighbors(path, name, v, showplot) :
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

        return math.floor(np.percentile(distancetrie,98))
    ```
    - "DBSCAN-STD" : Runs DBSCAN after standardizing the dataset. 


### Utility functions:
    -> search_k(path, name, showplot) : Select an appropriate number of clusters for k-means thanks to the elbow method
    -> distanceToNeighbors(path, name, v, showplot) : Finds an appropriate epsilon for DBSCAN by analyzing the average distance of points to their neighbors
    -> evaluate_runtime_DBSCAN(path, name) : Used to plot the runtime of multiple runs of DBSCAN with different values of epsilon.

The results are written onto results.csv 
