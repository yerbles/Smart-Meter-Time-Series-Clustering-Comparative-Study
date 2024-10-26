# Module Description
# These functions allow the computation of alternative data representations and their corresponding distance matrices

# *******************
# Module Requirements
# *******************

# General
from functools import partial
import numpy as np
from comparativestudycode.distances import distanceMatrix
import os
import sys

# PCA
from sklearn.decomposition import PCA

# PAA
from pyts.approximation import PiecewiseAggregateApproximation

# SAX
from pyts.approximation import SymbolicAggregateApproximation
import scipy
import string
from functools import partial
from pylcs import lcs
import Levenshtein

# BOF
from sklearn import preprocessing
import warnings
import tsfel
import tsfresh.feature_extraction.feature_calculators as tsfc
import antropy as ant
import nolds
import statsmodels.tsa.stattools as stm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statistics import mean

# DWT
from scipy.interpolate import interp1d
import pywt

# MODWT
from comparativestudycode.MODWT import modwt

# GAF and MTF
from pyts.image import GramianAngularField, MarkovTransitionField

# *******************
# High Level Function
# *******************
def distanceMatrix_Representations(data, acronym, **method_params):
    """
    This wrapper function provides a consistent way to get distance matrices from representations with various parameters.
    The function will return a pairwise distance matrix for the provided data computed using 
    the representation method referred to by the acronym (see table from paper).
    The provided data will be the raw time series.

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in rows.

    acronym: string
        Accepted Representation Codes: BOF, DWT, GAF, LSTM, MTF, MODWT, PAA, PCA, SAX

    method_params (using the same parameters from the corresponding external functions):
        --- n_components: integer
            Required for repPCA, repBOF
        --- feature_normalisation: boolean
            Required for repBOF
        --- window_size: integer
            Required for repPAA
        --- wavelet: string
            Required for repDWT, repMODWT
        --- level: integer
            Required for repDWT, repMODWT
        --- coeff_format: string
            Required for repDWT, repMODWT
        --- nbins: integer
            Required for repSAX, repMTF
        --- strategy: string
            Required for repSAX, repMTF
        --- distance: string
            Required for repSAX
        --- img_sz: integer
            Required for repGAF, repMTF
        --- method: string
            Required for repGAF

    Output
    ------
    dist_matrix: array (shape = (num_ts, num_ts))

    """

    rep_distance_functions = {
        "PCA": partial(repPCA,
            n_components = method_params.get("n_components", -1), 
            return_representation = False
            ),
        "PAA": partial(repPAA,
            window_size = method_params.get("window_size", -1), 
            return_representation = False
            ),
        "BOF": partial(repBOF,
            n_components = method_params.get("n_components", -1), 
            feature_normalisation = method_params.get("feature_normalisation", -1), 
            return_representation = False
            ),
        "DWT": partial(repDWT,
            level = method_params.get("level", -1), 
            coeff_format = method_params.get("coeff_format", -1), 
            mother_wavelet = method_params.get("wavelet", -1), 
            n_interp_points = 64, 
            return_representation = False
            ),
        "MODWT": partial(repMODWT,
            level = method_params.get("level", -1), 
            coeff_format = method_params.get("coeff_format", -1),
            mother_wavelet = method_params.get("wavelet", -1), 
            return_representation = False
            ),
        "SAX": partial(repSAX,
            nbins = method_params.get("nbins", -1), 
            strategy = method_params.get("strategy", -1), 
            distance = method_params.get("distance", -1), 
            return_representation = False
            ),
        "GAF": partial(repGAF,
            img_sz = method_params.get("img_sz", -1), 
            method = method_params.get("method", -1), 
            return_representation = False, 
            return_distances = True
            ),
        "MTF": partial(repMTF,
            img_sz = method_params.get("img_sz", -1), 
            nbins = method_params.get("nbins", -1), 
            strategy = method_params.get("strategy", -1), 
            return_representation = False, 
            return_distances = True
            ),
        # "LSTM": partial(),
    }

    func = rep_distance_functions[acronym]
    return func(data)


########################
# Features and Distances
########################

# ****************************
# Principal Component Analysis
# ****************************
def repPCA(data, n_components, return_representation = False):
    """
    A consistent wrapper for PCA().fit_transform() and dist_matrix computation

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series or their alternative representations stored in the rows.

    n_components: integer
        The number of components retained in the representation (retained in order of variance explained).

    return_representation: boolean, default=False
        Whether to also return the representation used to produce the distance matrix

    Output
    ------
    dist_matrix: array (shape=(num_ts, num_ts))

    features: array (shape=(num_ts, n_components))
        Unless num_ts<num_features, then features will have shape=(num_ts, num_ts).

    """
    features = PCA(n_components = n_components).fit_transform(data)
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation:
        return (dist_matrix, features)
    else:
        return dist_matrix 


# *********************************
# Piecewise Aggregate Approximation
# *********************************
def repPAA(data, window_size=1, return_representation = False):
    """
    A consistent wrapper for PiecewiseAggregateApproximation().transform() with dist_matrix computation

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    window_size: integer or float default=1
        Length of the sliding window. If float, it represents a percentage of the size of each time series and must be between 0 and 1.

    return_representation: boolean, default=False
        Whether to also return the representation used to produce the distance matrix

    Output
    ------
    dist_matrix: array (shape=(num_ts, num_ts))

    features: array (shape=(num_ts, new_num_features))

    """

    features = PiecewiseAggregateApproximation(window_size=window_size).transform(data)
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation:
        return (dist_matrix, features)
    else:
        return dist_matrix 



# ***************
# Bag of Features
# ***************
def repBOF(data, n_components, feature_normalisation, return_representation = False):
    """
    Computation of distance matrix from principal components after computation of a set of 76 varied time series features.

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    n_components: integer
        The number of components retained in the representation (retained in order of variance explained).

    feature_normalisation: boolean
        Whether to apply Min-Max normalisation to the raw features before PCA or not

    return_representation: boolean, default=False
        Whether to also return the representation used to produce the distance matrix

    Output
    ------
    dist_matrix: array (shape=(num_ts, num_ts))

    features: array (shape=(num_ts, n_components))
        Unless num_ts<num_features, then features will have shape=(num_ts, num_ts).

    """
    # Compute Features
    num_ts = data.shape[0]
    raw_features = np.zeros((num_ts, 76))
    for i in range(num_ts):
        raw_features[i,:] = computeFeatures(data[i,:])

    # Normalisation
    if feature_normalisation:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(raw_features)
    else:
        features = np.copy(raw_features)

    return repPCA(features, n_components, return_representation)

def computeFeatures(ts):
    """
    Computation of 76 features for a given time series x stored as a numpy array
    """
    fs_list = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        #antropy - 8
        fs_list.append(ant.hjorth_params(ts)[1])
        fs_list.append(ant.higuchi_fd(ts))
        fs_list.append(ant.num_zerocross(ts))
        fs_list.append(ant.hjorth_params(ts)[0])
        fs_list.append(ant.perm_entropy(ts))
        fs_list.append(ant.petrosian_fd(ts))
        fs_list.append(ant.katz_fd(ts))
        fs_list.append(ant.svd_entropy(ts))

        #nolds - 4
        fs_list.append(nolds.dfa(ts))
        fs_list.append(max(nolds.lyap_e(ts)).item())
#    with warnings.catch_warnings():
#        warnings.simplefilter('ignore')
        fs_list.append(nolds.corr_dim(ts,emb_dim = 3))
        fs_list.append(nolds.hurst_rs(ts).item())

        #TSFEL - 44
        num_TSFEL_fs = 44
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "TSFEL_subset.json")
        cfg_file = tsfel.get_features_by_domain("subset",json_path=json_path)
        TSFEL_fs = tsfel.time_series_features_extractor(cfg_file, ts, fs=len(ts), verbose=0).values[0]
        for i in range(num_TSFEL_fs):
            fs_list.append(TSFEL_fs[i])

        #tsfresh - 16
        fs_list.append(tsfc.augmented_dickey_fuller(ts,param=[{"attr":'teststat', "autolag":'AIC'}])[0][1])
        fs_list.append(tsfc.benford_correlation(ts))
        fs_list.append(tsfc.c3(ts,lag=4))
        fs_list.append(tsfc.count_above_mean(ts))
        fs_list.append(tsfc.longest_strike_above_mean(ts))
        fs_list.append(tsfc.longest_strike_above_mean(ts))
        fs_list.append(tsfc.time_reversal_asymmetry_statistic(ts,lag=3))
        fs_list.append(tsfc.cid_ce(ts,normalize=False))
        fs_list.append(tsfc.count_below_mean(ts))
        fs_list.append(tsfc.ratio_value_number_to_time_series_length(ts))
        fs_list.append(tsfc.quantile(ts, .25))
        fs_list.append(tsfc.quantile(ts, .75))
        fs_list.append(tsfc.percentage_of_reoccurring_datapoints_to_all_datapoints(ts))
        fs_list.append(tsfc.percentage_of_reoccurring_values_to_all_values(ts))
        fs_list.append(tsfc.mean_second_derivative_central(ts))
        fs_list.append(tsfc.mean_change(ts))

        #Other - 4
        fs_list.append(SimpleExpSmoothing(ts,initialization_method='estimated').fit().params['smoothing_level'])
        fs_list.append(np.argmax(ts))
#    with warnings.catch_warnings():
#        warnings.simplefilter('ignore')
        fs_list.append(stm.kpss(ts,regression='c')[0])
        fs_list.append(best_fit_slope(ts))

    return fs_list

def best_fit_slope(x):
    y = np.array(list(range(len(x))))+1
    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)**2) - mean(x**2)))
    return m

def returnFeatureNames():
    antropy_names = ['Hjorth Complexity', 'Higuchi Fractal Dimension',
                    'Number of Zero Crossings', 'Hjorth Mobility', 'Permutation Entropy',
                    'Petrosian Fractal Dimension', 'Katz Fractal Dimension', 'SVD Entropy']

    nolds_names = ['Detrended Fluctuation Analysis','Maximum Lyapunov Exponent','Correlation Dimension',
                   'Hurst Exponent']

    TSFEL_names = ['Total Energy','Area Under the Curve', 'Positive Turning Points', 'Peak to Peak Distance',
             'Absolute Energy','Autocorrelation','Centroid','Entropy','Human Range Energy',
             'Interquartile Range','Kurtosis','Skewness','Mean', 'Variance','Wavelet Entropy',
             'Zero Crossing Rate','Wavelet Absolute Mean','Max','Maximum Frequency','Max Power Spectrum',
             'Mean Absolute Deviation','Mean Absolute Differences','Median','Median Absolute Deviation',
             'Median Absolute Differences','Median Frequency','Min','Negative Turning Points',
             'Power Bandwidth','Root Mean Square','Signal Distance','Slope','Spectral Centroid',
             'Spectral Decrease','Spectral Distance','Spectral Entropy','Spectral Kurtosis',
             'Spectral Positive Turning Points','Spectral Roll-Off','Spectral Skewness','Spectral Slope',
             'Spectral Spread','Spectral Variation','Sum Absolute Differences']

    tsfresh_names = ['Augmented Dicky Fuller Test Statistic','Benford Correlation','C3 Statistic (Lag=4)',
                     'Count Above Mean','Longest Streak Above Mean','Longest Streak Below Mean',
                     'Time Reversal Asymmetry Statistic (Lag=3)','(CID) Complexity Invariant Distance',
                     'Count Below Mean','Ratio Value Number to Time Series Length', '25th Percentile',
                     '75th Percentile','Percentage of Reoccuring Data Points to All Data Points',
                     'Percentage of Reoccurring Values to All Values','Mean Second Derivative Central',
                     'Mean Change']

    Other_names = ['Simple Exponential Smoothing Parameter Alpha',
                  'Time of Maximum Value','KPSS Test Statistic',
                  'Linear Trend Timewise']

    feature_list = antropy_names + nolds_names + Other_names + TSFEL_names + tsfresh_names

    return feature_list



# **************************
# Discrete Wavelet Transform
# **************************
def repDWT(data, level, coeff_format, mother_wavelet, n_interp_points = 64, return_representation = False):
    """
    Computes a linear interpolation of the data and then computes DWT features up to the requested level with the given mother wavelet.
    Returns the distance matrix computed from coefficients in one of three formats described below

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    level: integer
        Decomposition level. For time series of length 64 with the haar wavelet the maximum is 6 (using pywt.dwt_max_level())

    coeff_format: string
        - "A": All 64 approximation and detail coefficients
        - "B": Approximation coefficients only
        - "C": Latest pair of approx and detail coefficients

    mother_wavelet: string, default="Haar"
        Wavelet to use

    n_interp_points: integer, default=64
        The number of points to use in the interpolatred time series, default is to use ther next power of 2

    return_representation: boolean, default=False
        Whether to also return the representation used to produce the distance matrix

    Output
    ------
    dist_matrix: array (shape=(num_ts, num_ts))

    features: array (shape=(num_ts, new_num_features))

    """
    # Interpolation
    x = np.linspace(0, 47, num = 48, endpoint = True)
    x_new = np.linspace(0, 47, num = n_interp_points, endpoint = True)
    f = interp1d(x, data, axis=1, kind="slinear")
    interpolated_data = f(x_new)

    # DWT
    wvt = pywt.Wavelet(mother_wavelet)
    coeff_matrix_list = pywt.wavedec(interpolated_data, wvt, level = level, axis = 1)

    # Coefficient Subset
    if coeff_format == "A":
        features = np.concatenate(coeff_matrix_list, axis=1)
    elif coeff_format == "B":
        features = coeff_matrix_list[0]
    elif coeff_format == "C":
        features = np.concatenate((coeff_matrix_list[0],coeff_matrix_list[1]), axis=1)

    # Distance Matrix
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation:
        return (dist_matrix, features)
    else:
        return dist_matrix 


# *******************
# Maximal Overlap DWT
# *******************
def repMODWT(data, level, coeff_format, mother_wavelet, return_representation = False):
    """
    Computes MODWT features up to the requested level with the given mother wavelet.
    Returns the distance matrix computed from coefficients in one of three formats described below

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    level: integer
        Decomposition level. For time series of length 64 with the haar wavelet the maximum is 6 (using pywt.dwt_max_level())

    coeff_format: string
        - "A": All 64 approximation and detail coefficients
        - "B": Approximation coefficients only
        - "C": Latest pair of approx and detail coefficients

    mother_wavelet: string, default="Haar"
        Wavelet to use

    return_representation: boolean, default=False
        Whether to also return the representation used to produce the distance matrix

    Output
    ------
    dist_matrix: array (shape=(num_ts, num_ts))

    features: array (shape=(num_ts, new_num_features))

    """
    num_ts = data.shape[0]
    coeff_matrix_list = [np.zeros((num_ts, 48)) for i in range(level+1)]

    for i in range(num_ts):
        coeffs = modwt(data[i,:], mother_wavelet, level)
        for j in range(level+1):
            coeff_matrix_list[j][i,:] = coeffs[j,:]

    # Coefficient Subset
    if coeff_format == "A":
        features = np.concatenate(coeff_matrix_list, axis=1)
    elif coeff_format == "B":
        features = coeff_matrix_list[0]
    elif coeff_format == "C":
        features = np.concatenate((coeff_matrix_list[0],coeff_matrix_list[1]), axis=1)

    # Distance Matrix
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation:
        return (dist_matrix, features)
    else:
        return dist_matrix 


# ********************************
# Symbolic Aggregate Approximation
# ********************************
def repSAX(data, nbins, strategy, distance, return_representation = False):
    """
    Compute the Symbolic Aggregate Approximation and use one to four different string distance measures to produce a dist_matrix

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    nbins: integer
        The number of bins to produce. It must be between 2 and min(n_timestamps, 26).

    strategy: string
        Strategy used to define the widths of the bins:
        - "uniform": All bins in each sample have identical widths
        - "quantile": All bins in each sample have the same number of points
        - "normal": Bin edges are quantiles from a standard normal distribution

    distance: string
        The distances to use when computing the dist_matrix
        - "MINDIST", "LCSS", "Lev" and "vLev"

    return_representation: boolean, default=False
        Whether to return the feature representation as well as the distance matrix

    Output
    ------
    features: array (shape=(num_ts, new_num_features))
        Reurned in a tuple with dist_matrices if return_representation==True.

    dist_matrix: array (shape=(num_ts, num_ts))
    """
    features_temp = SymbolicAggregateApproximation(n_bins = nbins, strategy = strategy).fit_transform(data)
    features = []
    for i in range(data.shape[0]):
        features.append("".join(features_temp[i,:]))

    if distance == "MINDIST":
        lookup_table = buildLookupTable(nbins)
        distance_function = partial(mindist, nbins=nbins, lookup_table = lookup_table)
    elif distance == "LCSS":
        distance_function = lcss
    elif distance == "Lev":
        distance_function = Levenshtein.distance
    elif distance == "vLev":
        distance_function = partial(vLev, nbins=nbins)
    else:
        print(f"{distance} is not supported by repSAX")

    dist_matrix = computeStringBasedDistanceMatrix(features, distance_function)

    if return_representation:
        return (dist_matrix, features)
    else:
        return dist_matrix

def computeStringBasedDistanceMatrix(data, distance_measure):
    """
    Given a method for computing the distance between two time series, this function will return a pairwise distance matrix using that method

    Parameters
    ----------
    data: list (length = num_ts)
        The time series stored ina  list as strings of length num_features

    distance_measure: callable
        A function that operates on two strings

    Output
    ------
    dist_matrix: array (shape = (num_ts, num_ts))
    
    """
    dist_matrix = np.zeros((len(data), len(data)))
    for i in range(0,len(data)-1,1):
        for j in range(i+1,len(data),1):
            dist_matrix[i,j] = distance_measure(data[i],data[j])
            dist_matrix[j,i] = dist_matrix[i,j]   

    return dist_matrix

def mindist(s1, s2, nbins, lookup_table):
    """
    Computes the MINDIST distance between two strings of equal length (s1 and s2)
    """
    length_ts = len(s1)
    alphabet = list(string.ascii_lowercase[0:nbins])

    inds1 = [alphabet.index(char) for char in s1]
    inds2 = [alphabet.index(char) for char in s2]

    values = np.zeros(length_ts)
    for i in range(length_ts):
        values[i] = lookup_table[inds1[i], inds2[i]]

    return np.sqrt(np.sum(values**2))

def buildLookupTable(nbins):
    """
    Utility function for MINDIST which only needs to be computed once for a given alphabet size.
    """
    break_points = scipy.stats.norm.ppf(np.linspace(0, 1, nbins + 1)[1:-1])
    lookup_table = np.zeros((nbins, nbins))

    for r in range(0,nbins):
        for c in range(r+2, nbins):
            lookup_table[r,c] = break_points[max([r,c])-1] - break_points[min([r,c])]
    lookup_table += lookup_table.T

    return lookup_table

def lcss(s1, s2):
    """
    Computes the longest common subsequence distance between two strings of equal length (s1 and s2)
    """
    lcs_length = lcs(s1,s2)

    return 1-np.abs(lcs_length)/len(s1)

def vLev(s1, s2, nbins):
    """
    Computes a variant of the Levenshtein distance between two strings of equal length (s1 and s2) enforcing an ordering on the alphabet
    """
    edits = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            edits += symbol_dist(s1[i],s2[i])

    return edits/((nbins-1)*len(s1))

def symbol_dist(char1, char2):
    """
    Utility function for vLev
    """
    alpahbetindex = {
    "a":1,
    "b":2,
    "c":3,
    "d":4,
    "e":5,
    "f":6,
    "g":7,
    "h":8,
    "i":9,
    "j":10,
    "k":11,
    "l":12,
    "m":13,
    "n":14,
    "o":15,
    "p":16,
    "q":17,
    "r":18,
    "s":19,
    "t":20,
    "u":21,
    "v":22,
    "w":23,
    "x":24,
    "y":25,
    "z":26}
    x = alpahbetindex[char1.lower()]
    y = alpahbetindex[char2.lower()]
    return max(x, y, key=float) - min(x, y, key=float)


def lcsv(X, Y,nbins):
    #lcs variant distance
    if len(X)!=len(Y):
        print('Error: X and Y should have the same length')
    else:
        edits = 0
        for i in range(len(X)):
            if X[i] != Y[i]:
                edits += symbol_dist(X[i],Y[i])
    return edits/((nbins-1)*len(X))


# ***********************
# Grammian Angular Field
# ***********************
def repGAF(data, img_sz, method, return_representation = False, return_distances = True):
    """
    Compute the Grammian Angular Field representation and compute distances using ED

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    img_sz: integer
        The size of the images, i.e. (img_sz x img_sz) which will be regarded as a flat array for distance computations

    method: string
        Either "summation" or "difference". See pyts documentation.

    return_representation: boolean, default=False
        Whether to return the feature representation as well as the distance matrix

    return_distances: boolean, default=True
        Whether to return the distance matrix computed with ED for the feature representation. At least one of this or return_representation must be True

    Output
    ------
    features: array (shape=(num_ts, new_num_features))
        Reurned in a tuple with dist_matrices if return_representation==True.

    dist_matrix: array (shape=(num_ts, num_ts))
    """

    if not return_representation and not return_distances:
        print("Must request either distances, features or both")

    features = GramianAngularField(image_size = img_sz, method = method, flatten = True).fit_transform(data)
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation and return_distances:
        return (dist_matrix, features)
    elif not return_representation:
        return dist_matrix
    else:
        return features


# ************************
# Markov Transition Field
# ************************
def repMTF(data, img_sz, nbins, strategy, return_representation = False, return_distances = True):
    """
    Compute the Grammian Angular Field representation and compute distances using ED

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series stored in the rows.

    img_sz: integer
        The size of the images, i.e. (img_sz x img_sz) which will be regarded as a flat array for distance computations

    nbins: integer
        Number of bins (also known as the size of the alphabet)

    strategy: string
        One of "uniform", "quantile" or "normal".

    return_representation: boolean, default=False
        Whether to return the feature representation as well as the distance matrix

    return_distances: boolean, default=True
        Whether to return the distance matrix computed with ED for the feature representation. At least one of this or return_representation must be True

    Output
    ------
    features: array (shape=(num_ts, new_num_features))
        Reurned in a tuple with dist_matrices if return_representation==True.

    dist_matrix: array (shape=(num_ts, num_ts))
    """

    if not return_representation and not return_distances:
        print("Must request either distances, features or both")

    features = MarkovTransitionField(image_size = img_sz, n_bins = nbins, strategy = strategy, flatten = True).fit_transform(data)
    dist_matrix = distanceMatrix(data = features, acronym = "ED")

    if return_representation and return_distances:
        return (dist_matrix, features)
    elif not return_representation:
        return dist_matrix
    else:
        return features




# Test code
if __name__ == "__main__":
    x = np.random.normal(size=(50,48)) + np.random.uniform(size=(50,48))
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    print("x.shape: ", x.shape)
    print("")

    # High-level function test
    for acro in ["PAA", "PCA", "BOF", "DWT", "MODWT", "SAX", "GAF", "MTF"]:
        print(f"{acro}: ", end="")
        d = distanceMatrix_Representations(
            x, acro,
            img_sz = 10, method="summation", nbins = 18, strategy = "normal", level = 5, coeff_format="A",
            n_components = 10, feature_normalisation=True, distance="MINDIST", window_size = 10,)
        print(d[0,1])
    print("")



    # # GAF
    # print("GAF")
    # dist_matrix, features = repGAF(x, img_sz = 10, type="summation", return_representation = True, return_distances = True)
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # print("")

    # # MTF
    # print("MTF")
    # dist_matrix, features = repMTF(x, img_sz = 10, nbins = 5, strategy = "normal", return_representation = True, return_distances = True)
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # print("")

    # # DWT
    # print("DWT")
    # dist_matrix, features = repDWT(x, level = 5, coeff_format="A", mother_wavelet = "haar", n_interp_points = 64, return_representation = True)
    # print("Format A")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # dist_matrix, features = repDWT(x, level = 5, coeff_format="B", mother_wavelet = "haar", n_interp_points = 64, return_representation = True)
    # print("Format B")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # dist_matrix, features = repDWT(x, level = 5, coeff_format="C", mother_wavelet = "haar", n_interp_points = 64, return_representation = True)
    # print("Format C")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)

    # # MODWT
    # print("MODWT")
    # dist_matrix, features = repMODWT(x, level = 5, coeff_format="A", mother_wavelet = "haar", return_representation = True)
    # print("Format A")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # dist_matrix, features = repMODWT(x, level = 5, coeff_format="B", mother_wavelet = "haar", return_representation = True)
    # print("Format B")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # dist_matrix, features = repMODWT(x, level = 5, coeff_format="C", mother_wavelet = "haar", return_representation = True)
    # print("Format C")
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)

    # # BOF
    # print("BOF")
    # dist_matrix, features = repBOF(x, n_components = 10, feature_normalisation=True, return_representation = True)
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # print("")

    # # SAX
    # print("SAX")
    # dist_matrices = repSAX(x, nbins=10, strategy="normal", distance="MINDIST")
    # print(dist_matrices.shape)
    # print("")

    # # PCA
    # print("PCA with 10 components")
    # dist_matrix, features = repPCA(x, n_components = 10, return_representation = True)
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # print("")

    # # # PAA
    # print("PAA with window size 10")
    # dist_matrix, features = repPAA(x, window_size = 10, return_representation = True)
    # print("features.shape: ", features.shape)
    # print("dist_matrix.shape: ", dist_matrix.shape)
    # print("")

