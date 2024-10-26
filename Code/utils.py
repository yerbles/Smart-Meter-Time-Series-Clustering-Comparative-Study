# Module Description
# Various functions that didn't belong elsewhere

# *******************
# Module Requirements
# *******************
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

def firstNearestNeighbours(dist_matrix, gt_labels):

    num_objects = len(gt_labels)
    if dist_matrix.shape[0] != num_objects:
        return "Error: Mismatch between shape of distance matrix and length of ground truth labels"

    nn = NearestNeighbors(n_neighbors = 1, algorithm='brute', metric="precomputed").fit(dist_matrix)
    nn_inds = nn.kneighbors()[1].reshape(-1)

    predicted = gt_labels[nn_inds]
    accuracy = np.sum(gt_labels == predicted)/num_objects

    return (accuracy, predicted)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
