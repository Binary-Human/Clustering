# Clustering
5SDBD - Apprentissage non-supervisé

## Prerequisites

### Before running

Dowload datasets via this link, to easily test the code

https://github.com/deric/clustering-benchmark

You'll find them in src/main/resources/datasets/artificial

### Virtual environnment Anaconda

```bash
conda create –name clustering-tp
conda activate clustering-tp 
conda info –envs
conda install numpy scipy matplotlib scikit-learn math kneed sys
conda list

```

## Running the code
```bash
    python3 ./extrait-code/archive-code/Find-best-solution.py "[Dataset]" "[Algorithm]"
```

### Different options

Most of what happens is through the find-best-solution script

- "KMEANS" : Runs k-means with the most adequate parameter for k, thanks to optimal_k_silhouette in Silhouette_optimal_k
- "AGLO" : something
- "DBSCAN" : Runs DBSCAN with the most adequate parameters for epsilon, thanks to distanceToNeighbors in Assessing_Neighbors
- "DBSCAN-STD" : Runs DBSCAN after standardizing the dataset. 


### Utility functions:

    - search_k(path, name, showplot) : Select an appropriate number of clusters for k-means thanks to the elbow method
    - optimal_k_silhouette(path, name, showplot) : Select an appropriate number of clusters for k-means by analysing the silhouette index [The one that is effectively used]
    - distanceToNeighbors(path, name, v, showplot) : Finds an appropriate epsilon for DBSCAN by analyzing the average distance of points to their neighbors
    - evaluate_runtime_DBSCAN(path, name) : Used to plot the runtime of multiple runs of DBSCAN with different values of epsilon.

**Exemple:**

```bash
    python3 ./extrait-code/archive-code/Evaluate_DBSCAN_runtime.py '3MC.arff'
    python3 ./extrait-code/archive-code/Assessing_Neighbors.py '3MC.arff' 5
    python3 ./extrait-code/archive-code/Searching_for_best_k.py '3MC.arff'
    python3 ./extrait-code/archive-code/Silhouette_optimal_k.py '3MC.arff'

```

The results are written onto results.csv 
