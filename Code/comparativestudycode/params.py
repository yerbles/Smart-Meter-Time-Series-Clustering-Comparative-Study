import numpy as np
import itertools
import string
#import pywt

def get_param_combinations(acronym, best=False):
    param_functions = {
        "BOF": get_bof_params,
        "DTW": get_dtw_params,
        "DWT": get_dwt_params,
        "ERP": get_erp_params,
        "ERS": get_ers_params,
        "GAF": get_gaf_params,
        "HMM": get_hmm_params,
        "KSD": get_ksd_params,
        "LSTM": get_lstm_params,
        "LSS": get_lss_params,
        "LCSS": get_lcss_params,
        "MPD": get_mpd_params,
        "MMP": get_mmp_params,
        "MSM": get_msm_params,
        "MTF": get_mtf_params,
        "MODWT": get_modwt_params,
        "PAA": get_paa_params,
        "PCA": get_pca_params,
        "SAX": get_sax_params,
        "TWED": get_twed_params
    }

    if acronym in param_functions:
        return param_functions[acronym](best)
    else:
        return [()]  # Return an empty tuple for metrics with no parameters

def get_all_param_combinations(acronym, best=False):
    params = get_param_combinations(acronym, best)

    if not params:
        return [()]

    # Check if the first item is already a tuple of tuples
    if isinstance(params[0], tuple) and all(isinstance(item, tuple) for item in params[0]):
        return params

    # For single parameter methods (like PAA)
    if isinstance(params[0], tuple) and len(params[0]) == 2:
        return params

    # For multi-parameter methods (like SAX, MTF, TWED)
    return [tuple(param_tuple) for param_tuple in params]

def get_param_names(acronym):
    param_names_dict = {
        "BOF": ["n_components", "feature_normalisation"],
        "DTW": ["window"],
        "DWT": ["wavelet", "level", "coeff_format"],
        "ERP": ["window", "g"],
        "ERS": ["window", "epsilon"],
        "GAF": ["img_sz", "method"],
        "HMM": ["n_states", "init"],
        "KSD": ["ksd_window"],
        "LSTM": ["architecture", "batch_size", "n_epochs"],
        "LSS": ["min_k"],
        "LCSS": ["window", "epsilon"],
        "MPD": ["m"],
        "MMP": ["p"],
        "MSM": ["window", "c"],
        "MTF": ["img_sz", "nbins", "strategy"],
        "MODWT": ["wavelet", "level", "coeff_format"],
        "PAA": ["window_size"],
        "PCA": ["n_components"],
        "SAX": ["distance", "nbins", "strategy"],
        "TWED": ["nu", "lmbda"]
    }
    return param_names_dict.get(acronym, [])

# Individual parameter functions
def get_bof_params(best=False):
    if best:
        return []
    n_comp_values = np.arange(5,80,5)
    feat_norm_options = [True, False]
    return list(itertools.product(
        [("n_components", n) for n in [76]+list(n_comp_values)],
        [("feature_normalisation", f) for f in feat_norm_options]
    ))

def get_dtw_params(best=False):
    if best:
        return [("window", w/48) for w in [1,2,3,4,5,6]]
    return [("window", w/48) for w in np.arange(1,49,1)]

def get_dwt_params(best=False):
    if best:
        return []
    n_points = 64
    wvt_strings = ["db1", "db2", "db3", "db4", "db5"]
    coeff_options = ["A", "B", "C"]

    params = []
    for wvt_string in wvt_strings:
        wvt = pywt.Wavelet(wvt_string)
        max_level = pywt.dwt_max_level(n_points, wvt)
        levels = np.arange(1, max_level+1, 1)

        params = params + list(itertools.product(
        [("wavelet", wvt_string)],
        [("level", l) for l in levels],
        [("coeff_format", c) for c in coeff_options]
    ))
    return params

def get_erp_params(best=False):
    if best:
        return list(itertools.product(
            [("window", w/48) for w in [1,2,3,4,5]],
            [("g", gap) for gap in [0,0.1,0.2,0.3]]
        ))
    window_values = np.arange(1,49,1)
    gap_values = np.arange(0,1.1,0.1)
    return list(itertools.product(
        [("window", w/48) for w in window_values],
        [("g", gap) for gap in gap_values]
    ))

def get_ers_params(best=False):
    if best:
        return []
    window_values = np.arange(1, 49, 1)
    eps_values = [None] + list(np.arange(0,1.1,0.1))
    return list(itertools.product(
        [("window", w/48) for w in window_values],
        [("epsilon", eps) for eps in eps_values]
    ))

def get_gaf_params(best=False):
    if best:
        return []
    img_sz_values = np.arange(2,49,1)
    method_options = ["summation", "difference"]
    return list(itertools.product(
        [("img_sz", i) for i in img_sz_values],
        [("method", m) for m in method_options]
    ))

def get_hmm_params(best=False):
    if best:
        return []
    state_values = np.arange(2,49,1)
    init_options = ["True", "False"]
    return list(itertools.product(
        [("n_states", s) for s in state_values],
        [("init", i) for i in init_options]
    ))

def get_ksd_params(best=False):
    if best:
        return [("ksd_window", w) for w in [1,2,3]]
    return [("ksd_window", w) for w in np.arange(1,49,1)]

def get_lstm_params(best=False):
    if best:
        return list(itertools.product(
        [("architecture", a) for a in string.ascii_uppercase[:12]],
        [("batch_size", b) for b in [10]],
        [("n_epochs", e) for e in [1000]]
    ))
    architectures = string.ascii_uppercase[:16]
    batch_size_values = [10,50,100]
    n_epochs_values = [100, 500, 1000]
    return list(itertools.product(
        [("architecture", a) for a in architectures],
        [("batch_size", b) for b in batch_size_values],
        [("n_epochs", e) for e in n_epochs_values]
    ))

def get_lss_params(best=False):
    if best:
        return []
    return [("min_k", min_k) for min_k in np.arange(44,49,1)]

def get_lcss_params(best=False):
    if best:
        return []
    window_values = np.arange(1, 49, 1)
    eps_values = np.arange(0,1.1,0.1)
    return list(itertools.product(
        [("window", w/48) for w in window_values],
        [("epsilon", gap) for gap in eps_values]
    ))

def get_mpd_params(best=False):
    if best:
        return [("m", m) for m in [41,42,43,44,45,46,47]]
    return [("m", m) for m in np.arange(3,49,1)]

def get_mmp_params(best=False):
    if best:
        return [("p", p) for p in [3,4,5]]
    return [("p", p) for p in [0.5,1,2,3,4,5]]

def get_msm_params(best=False):
    if best:
        return list(itertools.product(
        [("window", w/48) for w in [1,2,3,4,5,6]],
        [("c", c) for c in [0.1]]
    ))
    window_values = np.arange(1, 49, 1)
    c_values = [0.01,0.1,1,10,100]
    return list(itertools.product(
        [("window", w/48) for w in window_values],
        [("c", c) for c in c_values]
    ))

def get_mtf_params(best=False):
    if best:
        return []
    img_sz_values = np.arange(2,49,1)
    nbins_values = np.arange(2,27,1)
    strategy_options = ["uniform", "quantile", "normal"]
    return list(itertools.product(
        [("img_sz", i) for i in img_sz_values],
        [("nbins", b) for b in nbins_values],
        [("strategy", s) for s in strategy_options]
    ))

def get_modwt_params(best=False):
    if best:
        return []
    n_points = 48
    wvt_strings = ["db1", "db2", "db3", "db4", "db5"]
    coeff_options = ["A", "B", "C"]

    params = []
    for wvt_string in wvt_strings:
        wvt = pywt.Wavelet(wvt_string)
        max_level = pywt.dwt_max_level(n_points, wvt)
        levels = np.arange(1, max_level+1, 1)

        params = params + list(itertools.product(
        [("wavelet", wvt_string)],
        [("level", l) for l in levels],
        [("coeff_format", c) for c in coeff_options]
    ))
    return params

def get_paa_params(best=False):
    if best:
        return []
    return [("window_size", w) for w in np.arange(1,25,1)]

def get_pca_params(best=False):
    if best:
        return [("n_components", n) for n in [5,6,7,8,9,10,11,12,13,14]]
    return [("n_components", n) for n in np.arange(1,49,1)]

def get_sax_params(best=False):
    if best:
        return []
    distance_options = ["MINDIST", "LCSS", "Lev", "vLev"]
    nbins_values = np.arange(2,27,1)
    strategy_options = ["uniform", "quantile", "normal"]
    return list(itertools.product(
        [("distance", i) for i in distance_options],
        [("nbins", b) for b in nbins_values],
        [("strategy", s) for s in strategy_options]
    ))

def get_twed_params(best=False):
    if best:
        return list(itertools.product(
        [("nu", nu) for nu in [0.0001,0.001,0.01]],
        [("lmbda", lmbda) for lmbda in [0.25,0.5,0.75]]
    ))
    nu_values = [10**(-i) for i in range(6)]
    lambda_values = np.arange(0, 1.25, 0.25)
    return list(itertools.product(
        [("nu", nu) for nu in nu_values],
        [("lmbda", lmbda) for lmbda in lambda_values]
    ))


# Example usage
if __name__ == "__main__":
    for method in ['BOF', 'DTW', 'DWT', 'ERP', 'ERS', 'GAF', 'HMM', 'KSD', 'LSTM', 'LSS', 'LCSS', 'MPD', 'MMP', 'MSM', 'MTF', 'MODWT', 'PAA', 'PCA', 'SAX', 'TWED']:
        print(f"\n{method} all parameter combinations:")
        params_ls = get_all_param_combinations(method)
        for params in params_ls[:3]:  # Print first 3 combinations
            print(params)
        print(f"{len(params_ls)} {method} parameter combinations")

        print(f"\n{method} best parameter combinations:")
        best_params_ls = get_all_param_combinations(method, best=True)
        for params in best_params_ls[:3]:  # Print first 3 combinations
            print(params)
        print(f"{len(best_params_ls)} {method} best parameter combinations")
