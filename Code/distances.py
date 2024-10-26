# Module Description
# These functions allow the computation of pairwise distance matrices according to each of the distance measures in the paper

# *******************
# Module Requirements
# *******************
from functools import partial
from math import sqrt
import numpy as np
from aeon.distances import (
    dtw_pairwise_distance,
    erp_pairwise_distance,
    edr_pairwise_distance,
    lcss_pairwise_distance,
    msm_pairwise_distance,
    sbd_pairwise_distance,
    twe_pairwise_distance
)
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.spatial.distance import directed_hausdorff, pdist, squareform
from stumpy import mpdist
from scipy.optimize import linear_sum_assignment


# ********************
# High Level Functions
# ********************
def distanceMatrix(data, acronym, **method_params):
    """
    This function will return a pairwise distance matrix for the provided data computed using 
    the distance measure referred to by the acronym (see table from paper).
    The provided data will either be the raw time series, or a feature representation - in which case the function will be called with ED.

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in rows.

    acronym: string
        Accepted Distance Codes: BD, CaD, ChD, CID, CoD, CaD, DTW, ERP, ERS, FD, HD, KSD, KT, LSS, LCSS, MD, MAH, MPD, MMP, MSM, PC, SBD, SC, TWED

    method_params (using the same parameters from the corresponding external functions):
        --- p: float
            The order of the norm of the difference, e.g. (2 for Euclidean, 1 for Manhattan) - required for MMP
        --- window: float
            As a percentage (0<=window<=1) of num_features, the window to use for the bounding matrix - required for DTW, ERP, ERS, LCSS, MSM, TWED
            Default is 1
        --- g: float
            The reference value to penalise gaps - required for ERP
            Default is 0
        --- epsilon: float
            Matching threshold to determine if two subsequences are considered close enough to be considered 'common' - required for LCSS, ERS
            Default is 
        --- c: float
            Cost for split or merge operation - required for MSM
        --- nu: float
            A non-negative constant which characterizes the stiffness of the elastic TWED measure, must be > 0 - required for TWED
        --- lmbda: float
            A constant penalty that punishes the editing efforts, must be >=1.0 - required for TWED
        --- min_k: integer
            The minimum subsequence length considered, must be <= num_features - required for LSS
        --- m: integer
            Window size, must be <= num_features - required for MPD
        --- ksd_window: integer
            Window size, must be <= num_features - required for KSD

    Output
    ------
    dist_matrix: array (shape = (num_ts, num_ts))

    """

    num_ts, num_features = data.shape

    if acronym in ["BD", "CaD", "ChD", "CoD", "ED", "MD", "MAH"]:
        dist_codes = {
            "BD": "braycurtis",
            "CaD": "canberra",
            "ChD": "chebyshev",
            "CoD": "cosine",
            "ED": "euclidean",
            "MD": "cityblock",
            "MAH": "mahalanobis"
            }
        dist_code = dist_codes[acronym]
        dist_matrix = squareform(pdist(data, metric=dist_code))

    elif acronym == "MMP":
        dist_matrix = squareform(pdist(data, metric="minkowski", p=method_params.get("p", 0)))

    elif acronym in ["DTW", "ERP", "ERS", "LCSS", "MSM", "SBD", "TWED"]:
        aeon_pairwise_distances = {
            "DTW": partial(dtw_pairwise_distance, window=method_params.get("window",1)), # Note that the tslearn, dtaidistance and pyts implementations return the sqrt(dist), while aeon returns dist.
            "ERP": partial(erp_pairwise_distance, window=method_params.get("window",1), g=method_params.get("g",0)),
            "ERS": partial(edr_pairwise_distance, window=method_params.get("window",1), epsilon=method_params.get("epsilon",None)),
            "LCSS": partial(lcss_pairwise_distance, window=method_params.get("window",1), epsilon=method_params.get("epsilon",1)),
            "MSM": partial(msm_pairwise_distance, window=method_params.get("window",1), c=method_params.get("c",1)),
            "SBD": partial(sbd_pairwise_distance, standardize=False), # Normalisation applied before distance computations
            "TWED": partial(twe_pairwise_distance, window=method_params.get("window",1), nu=method_params.get("nu",0.001), lmbda=method_params.get("lmbda",1.0))
        }
        aeon_pairwise_distance = aeon_pairwise_distances[acronym]
        dist_matrix = aeon_pairwise_distance(data)
        if acronym == "DTW":
            dist_matrix = np.sqrt(dist_matrix)

    elif acronym in ["PC", "SC", "KT", "CID", "HD", "LSS", "MPD", "FD", "KSD"]:
        misc_distance_measures = {
            "PC": pearsonMetric,
            "SC": spearmanMetric,
            "KT": kendallMetric,
            "CID": complexityInvariantDistance,
            "HD": hausdorffDistance,
            "LSS": partial(LSS, min_k = method_params.get("min_k",None)),
            "MPD": partial(mpdist, m=method_params.get("m",None), normalize=False, p=2),
            "FD": flexDistance,
            "KSD": partial(kSlidingDistance, k=method_params.get("ksd_window", None))
        }
        distance_measure = misc_distance_measures[acronym]
        dist_matrix = computeDistanceMatrix(data=data, distance_measure=distance_measure)

    return dist_matrix



def computeDistanceMatrix(data, distance_measure):
    """
    Given a method for computing the distance between two time series, this function will return a pairwise distance matrix using that method

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in rows.

    distance_measure: callable
        A function that operates on 1 dimensional numpy arrays and returns a float.

    Output
    ------
    dist_matrix: array (shape = (num_ts, num_ts))
    
    """

    dist_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(0,data.shape[0]-1,1):
        for j in range(i+1,data.shape[0],1):
            dist_matrix[i,j] = distance_measure(data[i,:],data[j,:])
            dist_matrix[j,i] = dist_matrix[i,j]   

    return dist_matrix


# *******************
# Low Level Functions
# *******************
def pearsonMetric(x,y):
    """
    Compute the pearson correlation metric presented in "Victor Solo. Pearson Distance is not a Distance. arXiv, 8 2019."
    i.e. sqrt(1-pearsonr)
    """
    return sqrt(1-pearsonr(x,y)[0])


def spearmanMetric(x,y):
    """
    Compute a spearman correlation metric analagous to the pearsonMetric()
    i.e. sqrt(1-spearmanr)
    """
    return sqrt(1-spearmanr(x,y)[0])


def kendallMetric(x,y):
    """
    Compute a kendall's tau correlation metric analagous to the pearsonMetric()
    i.e. sqrt(1-kendalltau)
    """
    return sqrt(1-kendalltau(x,y)[0])


def complexityInvariantDistance(x,y):
    """
    Compute the complexity invariant distance for two 1d arrays
    """
    ce_x = np.sqrt(np.sum(np.diff(x)**2)) # ce: complexity estimate
    ce_y = np.sqrt(np.sum(np.diff(y)**2))
    return np.linalg.norm(x-y)*(max([ce_x, ce_y])/min(ce_x, ce_y))


def hausdorffDistance(x,y):
    """
    Compute the hausdorff distance for two 1d arrays
    """
    u = x.reshape(-1,1)
    v = y.reshape(-1,1)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])


def LSS(x,y,min_k):
    """
    Compute the local shape based similarity for two 1d arrays
    min_k is the smallest sequence length considered
    Equation 20 from: "On the selection of appropriate distances for gene expression data clustering"
    """
    n = len(x)
    if min_k > n:
        print("Error: min_k must be less than or equal to the len(x)")
    similarity = []
    for k in range(min_k,n+1,1):
        similarity.append(sim_k(x,y,k,n))
    return 1-np.max(similarity)

def sim_k(x,y,k,n):
    """
    A sub function for the local shape based similarity: LSS()
    Equation 21 from: "On the selection of appropriate distances for gene expression data clustering"
    """
    S = []
    for i in range(1,n+1-k+1):
        for j in range(1,n+1-k+1):
            SP = spearmanr(x[i-1:i-1+k],y[j-1:j-1+k])[0]
            S.append(SP)
    return np.max(S)


def flexDistance(series_a, series_b, *weight_matrix):
    """
    Provided by Rui Yuan, author of " A New Time Series Similarity Measure and Its Smart Grid Applications"

    By default, the time series series_a and series_b should be the same length 1d arrays
    Weight matrix is a tuple and is optional
    """
    sliding = series_a.shape[0]
    if weight_matrix:
        weight_amp_metrix = weight_matrix[0]
        weight_tem_metrix = weight_matrix[1]
    else:
        scale = max(series_a)-min(series_a) if max(series_a)-min(series_a) > 0 else 1
        weight_amp_metrix = np.ones((sliding,sliding))*sliding
        weight_tem_metrix = np.ones((sliding,sliding))*scale
    # build the merged matrix for FD
    amplitude_matrix = np.zeros((sliding, sliding))
    temporal_matrix = np.zeros((sliding, sliding))
    for i in range(sliding):
        for j in range(sliding):
            amplitude_matrix[i][j] = series_a[i]-series_b[j]
            temporal_matrix[i][j] = i-j
    merged_matrix = np.abs(amplitude_matrix)*weight_amp_metrix + np.abs(temporal_matrix)*weight_tem_metrix
    # build the distance
    row_ind, col_ind = linear_sum_assignment(merged_matrix)
    # return merged_matrix[row_ind, col_ind].mean(), row_ind, col_ind if you want to check the selected route represented by row and col
    return merged_matrix[row_ind, col_ind].mean()


def kSlidingDistance(x,y,k):
    """
    The k-sliding distance between equal length 1d numpy arrays x and y using a window of size k
    """
    ksd = 0
    shift_indices = np.arange(-k,k+1)
    valid_indices = np.arange(0,len(x),1)
    for i in range(len(x)):
        vals_x = []
        vals_y = []
        for s in shift_indices[np.isin(shift_indices+i,valid_indices)]+i:  # Ensures indices respect low and high boundaries
            vals_x.append(abs(x[i]-y[s]))
            vals_y.append(abs(y[i]-x[s]))
        vals = [np.min(vals_x), np.min(vals_y)]
        ksd += (np.max(vals))**2

    return sqrt(ksd)



# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------


# Test code
if __name__ == "__main__":
    x = np.random.normal(size=(10,10))
    print("x[0,:]: ", x[0,:])
    print("x[1,:]: ", x[1,:])
    print("")

    # sklearn
    for acro in ["BD", "CaD", "ChD", "CoD", "ED", "MD", "MAH"]:
        print(f"{acro}: ", end="")
        d = distanceMatrix(x, acro)
        print(d[0,1])
    print("")

    # minkowski
    for p in [0.5,3,4,5]:
        print(f"MM{p}: ", end="")
        d = distanceMatrix(x, "MMP", p = p)
        print(d[0,1])
    print("")

    # aeon
    for acro in ["DTW", "ERP", "ERS", "LCSS", "MSM", "SBD", "TWED"]:
        print(f"{acro}: ", end="")
        d = distanceMatrix(x, acro, window=0.05, g=0, epsilon=0.25, c=0.1, standardize=False, nu=0.001, lmbda=1.0)
        print(d[0,1])
    print("")

    # misc
    for acro in ["PC", "SC", "KT", "CID", "HD", "LSS", "FD", "KSD", "MPD"]:
        print(f"{acro}: ", end="")
        d = distanceMatrix(x, acro, min_k=8, m=9, ksd_window=5)
        print(d[0,1])
    print("")
