# Module Description
# Some quality of life functions for using various clustering algorithms

# *******************
# Module Requirements
# *******************
import numpy as np
from sklearn.cluster import Birch, SpectralClustering
import genieclust
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
import fastcluster
from scipy.cluster.hierarchy import fcluster

# ******
# BIRCH
# ******
def clusterBIRCH(dist_matrix, n_clusters, threshold=0.1, max_iters=10):
    recovered_n_clusters = 0
    iters = 0
    while (recovered_n_clusters < n_clusters):
        cluster_labels = Birch(n_clusters = n_clusters, threshold = threshold).fit_predict(dist_matrix)
        recovered_n_clusters = len(np.unique(cluster_labels))
        threshold /= 10
        iters += 1
        if iters == max_iters:
            return None
    return cluster_labels


# **********
# Genieclust
# **********
def clusterGC(dist_matrix, n_clusters, threshold=0.3):
    cluster_labels = genieclust.Genie(n_clusters=n_clusters, affinity="precomputed", gini_threshold=threshold).fit_predict(dist_matrix)
    return cluster_labels


# ********
# Spectral
# ********
def clusterSC(dist_matrix, n_clusters, delta=20, max_iters=20):
    recovered_n_clusters = 0
    iters = 0
    while recovered_n_clusters < n_clusters:
        cluster_labels = SpectralClustering(n_clusters=n_clusters, affinity="precomputed").fit_predict(np.exp(- dist_matrix ** 2 / (2. * delta** 2)))
        recovered_n_clusters = len(np.unique(cluster_labels))
        delta += 20
        iters += 1
        if iters == max_iters:
            return None
    return cluster_labels


# *********
# k-medoids
# *********
def clusterKMd(dist_matrix, n_clusters, n_inits=30, max_iters=200, verbose=False):
    instances = []
    min_instance = KMedoids(n_clusters=n_clusters, metric="precomputed", method="pam", init="k-medoids++", max_iter = max_iters).fit(dist_matrix)
    min_inertia = min_instance.inertia_
    count = 1
    if verbose:
        print(f"--- First initialisation with k-medoids++ \nInertia at init {count}: {min_inertia}\n--- Subsequent initialisations are random")
    for n in range(n_inits-1):
        new_instance = KMedoids(n_clusters=n_clusters, metric="precomputed", method="pam", init="random", max_iter = max_iters).fit(dist_matrix)
        if new_instance.inertia_ < min_inertia:
            min_instance = new_instance
            min_inertia = new_instance.inertia_
        count += 1
        if verbose:
            print(f"Best Inertia at init {count}: {min_inertia}")

    return min_instance.labels_


# *******
# k-means
# *******
def clusterKMn(data, n_clusters, n_inits=30, max_iters=200, tolerance=10e-5):
    # Consider using TimeSeriesKMeans from aeon toolkit which accepts callables for the distance function.
    cluster_labels = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_inits, max_iter=max_iters, tol=tolerance).fit_predict(data)
    return cluster_labels


# ***********************
# Hierarchical Clustering
# ***********************
def clusterHAC(dist_matrix, n_clusters, linkage_str):
    # linkage_str in ["single", "complete", "average", "weighted", "ward"]

    Z0 = squareform(dist_matrix)
    Z1 = fastcluster.linkage(Z0, method=linkage_str)
    cluster_labels = fcluster(Z1, t=n_clusters, criterion='maxclust')-1

    return cluster_labels



# Test code
if __name__ == "__main__":
	from sklearn.datasets import load_iris
	from sklearn.metrics import adjusted_rand_score
	from scipy.spatial.distance import pdist
	data = load_iris()
	X = data.data
	y = data.target
	print(X.shape)
	print(y.shape)
	dist_matrix = squareform(pdist(X))

	# BIRCH
	cluster_labels = clusterBIRCH(dist_matrix, n_clusters=4, threshold=0.1, max_iters=10)
	print("BIRCH: ", adjusted_rand_score(y,cluster_labels))

	# GC
	cluster_labels = clusterGC(dist_matrix, n_clusters=4, threshold=0.3)
	print("GC: ", adjusted_rand_score(y,cluster_labels))

	# SC
	cluster_labels = clusterSC(dist_matrix, n_clusters=4, delta=20, max_iters=20)
	print("SC: ", adjusted_rand_score(y,cluster_labels))

	# KMd
	cluster_labels = clusterKMd(dist_matrix, n_clusters=4, n_inits=30, max_iters=200, verbose=False)
	print("KMd: ", adjusted_rand_score(y,cluster_labels))

	# KMn
	cluster_labels = clusterKMn(X, n_clusters=4, n_inits=30, max_iters=200, tolerance=10e-5)
	print("KMn: ", adjusted_rand_score(y,cluster_labels))

	# HAC-C
	cluster_labels = clusterHAC(dist_matrix, n_clusters=4, linkage_str="complete")
	print("HAC-C: ", adjusted_rand_score(y,cluster_labels))
