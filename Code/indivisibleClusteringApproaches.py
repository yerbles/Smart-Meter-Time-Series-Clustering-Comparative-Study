# Module Description
# Some quality of life functions for using various indivisible clustering approaches

# *******************
# Module Requirements
# *******************
import numpy as np
from tslearn.clustering import KShape
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()
mixHMM = importr('mixHMM')

# *******
# k-shape
# *******
def approachKShape(data, n_clusters, n_inits=30, max_iter=200):
    # Data should have shape (num_ts, num_feats)
    ks = KShape(n_clusters, n_init=n_inits, max_iter=max_iter, init='random', tol=10e-5).fit(np.expand_dims(data, axis=2))
    return ks.labels_

# ****
# HMMs
# ****
def approachHMM(data, n_clusters, n_states, init, variance_type="heteroskedastic", n_tries=30, max_iter=200, verbose=False):
    # Convert numpy array to R matrix
    r_data = ro.r.matrix(data, nrow=data.shape[0], byrow=True)

    # Call the emMixHMM function
    result = mixHMM.emMixHMM(Y=r_data, K=n_clusters, R=n_states, variance_type=variance_type, n_tries=n_tries, max_iter=max_iter, threshold=10e-5, verbose=verbose)

    # Extract the cluster labels
    subset_func = ro.r("""function(rs4_obj){
    	rs4_obj$stat$klas
    	}
    """)

    return np.array(subset_func(result), dtype="int64").reshape(-1)




# Test code
if __name__ == "__main__":
    from aeon.datasets import load_arrow_head
    from sklearn.metrics import adjusted_rand_score
    import numpy as np

    X, y = load_arrow_head()
    X = X.reshape(211, -1)
    print("Data shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Unique labels:", np.unique(y))


    # K-shape
    cluster_labels = approachKShape(X,3,n_inits=1,max_iter=10)
    print("kshape: ", adjusted_rand_score(y,cluster_labels))

    # HMMs
    cluster_labels = approachHMM(synth_data, 20, n_states=3, n_tries=5, max_iter=100, init=True, verbose=True)
    print("HMM:", adjusted_rand_score(y, cluster_labels))
