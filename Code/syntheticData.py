# A range of functions used in the production of synthetic load curve data

# Load Modules
import numpy as np
from scipy.stats import norm, logistic
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#####################
# Top Level Functions
#####################

def syntheticDatasetGenerator(labels, counts, sigma = (0.12, 0.1), theta = (-0.1, 0.5), return_precursors = False, num_outliers = None, offset=0):
    """
    This function produces the synthetic dataset based on the provided labels, counts and parameters.
    Default model parameters provide a balance between realistic variation and certainty of cluster membership.

    Parameters
    ----------
    labels: list
        A list of labels identifying the clusters from which time series will be generated

    counts: list
        Corresponding counts for each label provided in labels

    sigma: tuple
        (sigma_low, sigma_high) - Variance of the white noise
        Changing sigma_low and sigma_high alters the white noise variance for for low (<0.5) and high (>=0.5) values of C_t respectively.

    theta: tuple
        (theta_low, theta_high) - Coefficients in the STAR model
        Interacts with the sigma values, but with all other parameters fixed, larger theta values smooth the load curves out, lower makes the peaks less prominent.

    num_outliers: integer
        The number of outliers to generate (these will have cluster label -1). Default is None.

    offset: integer
        A parameter used to control the similarity between the following pairs of clusters: 100 and 101, 200 and 201, 300 and 301.
        These are not included in the 20 base clusters, but are used instead for cluster morphing experiments.
        For 100 and 101: offset controls the difference in timing of the two peaks (integer from 0 to 200 in steps of 1)
        For 200 and 201: offset controls the relative height of the second peak (float from 1.0 to 0 in steps of 0.01)
        For 300 and 301: offset controls the difference in width of the two normal pdf peaks (float from 0 to 200 in steps of 1)

    return_precursors: boolean
        If true, function also returns the corresponding characteristic curves and long time series

    Returns
    -------
        array (shape = (num_ts, 48))
    """

    if num_outliers:
        cluster_labels = np.concatenate((np.repeat(labels,counts), -np.ones(num_outliers, dtype="int64")))
    else:
        cluster_labels = np.repeat(labels,counts)

    # Parameters
    num_points = 480
    num_ts = len(cluster_labels)
    sigma_low, sigma_high = sigma
    theta_low, theta_high = theta

    # Sequence Generation Loop
    prelim_synth_data = np.zeros((num_ts, num_points))
    subsampled_synth_data = np.zeros((num_ts, 48))

    if return_precursors:
        C_t_all = np.zeros((num_ts, num_points))

    for i, cluster_index in enumerate(cluster_labels):

            # Characteristic Curve
            C_t = C_func(cluster_index, offset)
            if return_precursors:
                C_t_all[i,:] = C_t

            # Initial Value
            if cluster_index not in [12,14]: # These two clusters do not start the day around zero
                prelim_synth_data[i,0] = norm.rvs(loc=0, scale=0.05, size=1)
            else:
                prelim_synth_data[i,0] = norm.rvs(loc=1.5, scale=0.05, size=1)

            # STAR Modelling
            for j in range(1,num_points,1):
                if C_t[j] < 0.5 and C_t[j] > 0:
                    W_t = norm.rvs(loc=0, scale=sigma_low, size=1)
                else:
                    W_t = norm.rvs(loc=0, scale=sigma_high, size=1)

                prelim_synth_data[i,j] = theta_low*prelim_synth_data[i,j-1] + (1 + theta_high*prelim_synth_data[i,j-1])*C_t[j] + W_t

            # Subsample every 10th point to gain 48 point daily load curves. The first index is random
            subsampled_synth_data[i,:] = prelim_synth_data[i,np.random.choice(np.arange(0,10,1))::10]

    # Normalisation
    final_synth_data = MinMaxScaler().fit_transform(subsampled_synth_data.T).T # The transposes ensure that each time series is normalised independently

    if return_precursors:
        return (final_synth_data, cluster_labels, C_t_all, prelim_synth_data)
    else:
        return (final_synth_data, cluster_labels)


def C_func(cluster_num, offset):
    # A reference function to pull the appropriate generator
    
    cluster_functions = {
        -1: lambda: C__Outliers(),
        0: lambda: C__NormalSinglePeak(location = 120, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        1: lambda: C__NormalSinglePeak(location = 190, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        2: lambda: C__NormalSinglePeak(location = 320, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        3: lambda: C__NormalSinglePeak(location = 390, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        4: lambda: C__CombineClusters(clusters = [0,2], mixture_ratios = [1,1]),
        5: lambda: C__CombineClusters(clusters = [0,3], mixture_ratios = [1,1]),
        6: lambda: C__CombineClusters(clusters = [0,1,2,3], mixture_ratios = [1,1,1,1]),
        7: lambda: C__CombineClusters(clusters = [0,3], mixture_ratios = [0.6,1]),
        8: lambda: C__CombineClusters(clusters = [0,3], mixture_ratios = [1,0.6]),
        9: lambda: C__NormalSinglePeak(location = 320, location_max_deviation = 10, width = 70, width_max_deviation = 5),
        10: lambda: C__LogisticSinglePeak(location = (120,260), location_max_deviation = (10,10), scale = (8,8), scale_max_deviation = (4,4)),
        11: lambda: C__LogisticSinglePeak(location = (320,460), location_max_deviation = (10,10), scale = (8,8), scale_max_deviation = (4,4)),
        12: lambda: C__LogisticSinglePeak(location = (None,60), location_max_deviation = (10,10), scale = (8,8), scale_max_deviation = (4,4)),
        13: lambda: C__LogisticSinglePeak(location = (420,None), location_max_deviation = (10,10), scale = (8,8), scale_max_deviation = (4,4)),
        14: lambda: C__LogisticMultiplePeaks(locations = [(None,60),(420,None)], location_max_deviation = (10,10), scale = (8,8), scale_max_deviation = (4,4)),
        15: lambda: C__NormalMultiplePeaks(locations = [160,220], location_max_deviation = 6, width = 13, width_max_deviation = 5),
        16: lambda: C__NormalMultiplePeaks(locations = [320,390], location_max_deviation = 8, width = 15, width_max_deviation = 5),
        17: lambda: C__SolarDipCluster(base_cluster = 7, location = 220, location_max_deviation = 4, width = 45, width_max_deviation = 4),
        18: lambda: C__SolarDipCluster(base_cluster = 3, location = 220, location_max_deviation = 4, width = 45, width_max_deviation = 4),
        19: lambda: C__LogisticSinglePeak(location = (315,395), location_max_deviation = (10,10), scale = (0.5,20), scale_max_deviation = (0,5)),
        #
        #
        # The following characteristic curves are used for the cluster morphing experiments.
        # The offset parameter is used to control the similarity between the pairs of clusters
        # 
        # ---------- Timing ----------
        100: lambda: C__NormalSinglePeak(location = 70, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        101: lambda: C__NormalSinglePeak(location = 70+offset, location_max_deviation = 10, width = 15, width_max_deviation = 5),
        #
        # ---------- Relative Magnitude ----------
        200: lambda: C__CombineClusters(clusters = [0,2], mixture_ratios = [1,1]),
        201: lambda: C__CombineClusters(clusters = [0,2], mixture_ratios = [1,offset]),
        #
        # ---------- Scale ----------
		# Need to explain this and Timing as a window of scales per point plotted, not all with the same width
        300: lambda: C__NormalSinglePeak(location = 240, location_max_deviation = 10, width = 50, width_max_deviation = 5),
        301: lambda: C__NormalSinglePeak(location = 240, location_max_deviation = 10, width = 50+offset, width_max_deviation = 5),
    }

    if cluster_num in cluster_functions:
        return cluster_functions[cluster_num]()
    else:
        return f"Invalid cluster_num {cluster_num}"

##############################################
# The Characteristic Curve Component Functions
##############################################

def randomParameterChoice(center, max_deviation):
    """
    Utility function for randomly selecting distribution parameters
    """
    return np.random.choice(np.arange(center-max_deviation, center+max_deviation+1, 1))


def C__NormalSinglePeak(location, location_max_deviation, width, width_max_deviation):
    """
    A function which returns characteristic curves with a single peak based on the normal pdf.
    The location and width can both be specified given num_points = 480.
    """
    C_t = norm.pdf(
        x = np.arange(0,480,1),
        loc = randomParameterChoice(location, location_max_deviation),
        scale = randomParameterChoice(width, width_max_deviation)
    )
    return C_t/np.max(C_t)


def C__NormalMultiplePeaks(locations, location_max_deviation, width, width_max_deviation):
    """
    A function which returns characteristic curves with multiple peaks each based on the normal pdf.
    The locations (list) and width can both be specified given num_points = 480.
    """
    C_t = C__NormalSinglePeak(locations[0], location_max_deviation, width, width_max_deviation)
    for location in locations[1:]:
        C_t = C_t + C__NormalSinglePeak(location, location_max_deviation, width, width_max_deviation)
    return C_t/np.max(C_t)


def C__NormalMultiplePeaksVariableHeights(locations, height_ratios, location_max_deviation, width, width_max_deviation):
    """
    A function which returns characteristic curves with multiple peaks each based on the normal pdf, and whose heights
    can be set relative to one another with height_ratios
    The locations (list) and width can both be specified given num_points = 480.
    """
    heights = height_ratios/np.max(height_ratios)
    order = np.flip(np.argsort(heights)) # Start with largest peak
    C_t = C__NormalSinglePeak(locations[order[0]], location_max_deviation, width, width_max_deviation)
    for index in order[1:]:
        C_t = C_t + heights[index]*C__NormalSinglePeak(locations[index], location_max_deviation, width, width_max_deviation)
    return C_t/np.max(C_t)


def C__LogisticSinglePeak(location, location_max_deviation, scale, scale_max_deviation):
    """
    A function which returns characteristic curves with a single peak based on a logistic CDF.
    This allows for steeper or ramp-up and ramp-down than the normal PDF.
    The location and width can both be specified as tuples, one for start and one for finish.
    Start (location[0]) or finish (location[1]) can be None if the profile should start elevated.
    """
    C_t = np.zeros(480)
    if location[0] is not None:
        C_t = C_t + logistic.cdf(
                x = np.arange(0,480,1), 
                loc = randomParameterChoice(location[0], location_max_deviation[0]), 
                scale = randomParameterChoice(scale[0], scale_max_deviation[0]), 
                )
    if location[1] is not None:
        C_t = C_t + np.abs(logistic.cdf(
                x = np.arange(0,480,1), 
                loc = randomParameterChoice(location[1], location_max_deviation[1]),  
                scale = randomParameterChoice(scale[1], scale_max_deviation[1]), 
            ) - 1)

    C_t = C_t - np.min(C_t)

    return C_t/np.max(C_t)


def C__LogisticMultiplePeaks(locations, location_max_deviation, scale, scale_max_deviation):
    """
    A function which returns characteristic curves with multiple peaks each based on the logistic CDF
    Tuples specify the start and end of each peak, with None being allowed to indicate a boundary peak.
    """
    C_t = np.zeros(480)
    for tup in locations:
        C_t = C_t + C__LogisticSinglePeak(location = tup, location_max_deviation = location_max_deviation, scale = scale, scale_max_deviation = scale_max_deviation)
    return C_t/np.max(C_t)


def C__SolarDipCluster(base_cluster, location, location_max_deviation, width, width_max_deviation):
    """
    Add a solar dip to any base cluster shape
    """
    C_t = C_func(base_cluster, offset=0) # Offset isn't used by any base clusters
    solar = norm.pdf(
        x = np.arange(0,480,1),
        loc = randomParameterChoice(location, location_max_deviation),
        scale = randomParameterChoice(width, width_max_deviation)
    )
    solar = (solar-np.min(solar))/np.max(solar)
    return C_t - 0.85*solar


def C__CombineClusters(clusters, mixture_ratios):
    """
    Combine any of the other clusters into a new one using ratios to control the relative height of their features
    clusters and mixture_ratios are both lists whose values should correspond.
    """
    C_t = np.zeros(480)
    for cluster, ratio in zip(clusters, mixture_ratios):
        C_t = C_t + ratio*C_func(cluster, offset=0) # Offset isn't used by any base clusters

    C_t = C_t - np.min(C_t)
    return C_t/np.max(C_t)


def probability_check(probability):
    return np.random.random() < probability

def C__Outliers():
    """
    Creates characteristic curves for outliers. 

    Notes
    -----
    - As many outlier identification methods rely on distance computations, it is necessary to produce outliers using an approach that is 
    independent of distance computations. Hence the output cannot be guaranteed to always produce time series that would be deemed outliers
    by all distance measures and representation methods.
    - The process is as follows:
        1. Produce a random number (between 1 and 10) of normal pdf peaks at random locations and with random scales
        2. Normalise and combine those peaks, normalise
        3. Add a single wide normal pdf peak with probability 4/5, normalise
        4. Add a solar dip with probability 1/5, normalise
        5. Add a randomly located step up or down using a logistic cdf to add level diversity, normalise
    """
    num_outlier_components = np.random.choice(np.arange(1,10,1)) # 5, 10
    C_t_ls = []
    locations = np.random.choice(np.arange(np.random.choice(np.arange(0,5,1)),480,5), size=num_outlier_components, replace=False) #np.random.choice(np.arange(0,480,1), size=num_outlier_components, replace=False)
    for i in range(num_outlier_components):
        C_t_ls.append(norm.pdf(
                        x = np.arange(0,480,1),
                        loc = locations[i],
                        scale = np.random.choice([2,3,4,5,6,7,8,9,10,11,12,13]) # 2 is the smallest consistently registerable scale after downsampling by 10
    ))
    C_t_ls = [C_t - np.min(C_t) for C_t in C_t_ls]
    C_t_ls = [C_t/np.max(C_t) for C_t in C_t_ls]

    C_t = np.sum(C_t_ls, axis=0)
    C_t = C_t - np.min(C_t)
    C_t = C_t/np.max(C_t) 

    # Single lower and wider peak added with a probability of 4/5
    if probability_check(0.8):
        location = np.random.choice(np.arange(0,480,1), size=1)
        width = np.random.choice([2,3,4,5,6,7,8,9,10,11,12,13])
        extra = norm.pdf(
                x = np.arange(0,480,1),
                loc = location,
                scale = np.random.choice(np.arange(50,100,1), size=1)
                )
        extra = (extra-np.min(extra))/np.max(extra)
        C_t = C_t + np.random.choice(np.arange(0.2,0.81,0.1),size=1)*extra

        C_t = C_t - np.min(C_t)
        C_t = C_t/np.max(C_t) 

    # Solar dip added with a probability of 1/5
    if probability_check(0.2):
        location = 220
        location_max_deviation = 4
        width = 45 
        width_max_deviation = 4
        solar = norm.pdf(
            x = np.arange(0,480,1),
            loc = randomParameterChoice(location, location_max_deviation),
            scale = randomParameterChoice(width, width_max_deviation)
            )
        solar = (solar-np.min(solar))/np.max(solar)
        C_t = C_t - 0.85*solar

    # Add a randomly located step up with probability 3/4
    if probability_check(0.75):
        location = np.random.choice(np.arange(1,479,1), size=1) # 1 in from each end so a step actually exists
        if probability_check(0.5):
            extra = logistic.cdf(
                        x = np.arange(0,480,1), 
                        loc = np.random.choice(np.arange(60,419,1), size=1),
                        scale = np.random.choice(np.arange(1,11,1), size=1), 
                        )
        else:
            extra = 1-logistic.cdf(
                        x = np.arange(0,480,1), 
                        loc = np.random.choice(np.arange(60,419,1), size=1),
                        scale = np.random.choice(np.arange(1,11,1), size=1), 
                        )
        extra = (extra-np.min(extra))/np.max(extra)

        C_t = C_t + np.random.choice(np.arange(0.2,0.81,0.1),size=1)*extra
        C_t = C_t - np.min(C_t)
        C_t = C_t/np.max(C_t) 

    return C_t 



# def C__Outliers():
#     """
#     """
#     num_outlier_components = np.random.choice(np.arange(1,25,1))
#     C_t = np.zeros(480)
#     locations = np.random.choice(np.arange(np.random.choice(np.arange(0,20,1)),480,20), size=num_outlier_components, replace=False)
#     for i in range(num_outlier_components):
#         C_t = C_t  + norm.pdf(
#                         x = np.arange(0,480,1),
#                         loc = locations[i],
#                         scale = np.random.uniform(10,20,1) # 2 is the smallest consistently registerable scale after downsampling by 10
#     )
#     C_t = C_t - np.min(C_t)
#     C_t = C_t/np.max(C_t) 

#     # Solar dip added with a probability of 1/5
#     if probability_check(0.2):
#         location = 220
#         location_max_deviation = 4
#         width = 45 
#         width_max_deviation = 4
#         solar = norm.pdf(
#             x = np.arange(0,480,1),
#             loc = randomParameterChoice(location, location_max_deviation),
#             scale = randomParameterChoice(width, width_max_deviation)
#             )
#         solar = (solar-np.min(solar))/np.max(solar)
#         C_t = C_t - 0.85*solar
    
#     return C_t 


# def C__Outliers(num_outlier_components = 8):
#     """
#     Produces outlier characteristic curves by summing existing characteristic curves together. The num_outlier_components (integer)
#     components are selected at random without replacement from clusters 0 to 20 (and including -1 allowing for recursion).

#     Parameters
#     ----------
#     num_outlier_constituents: integer
#         Outliers are generated by combining (via multiplication) characteristic curves from existing clusters. This parameter specifies the number of 
#         constituent characteristic curves which will be selected at random without replacement. Default is None.
#     """
#     if num_outlier_components is None:
#         return "Error: please specify num_outlier_components when calling C__Outliers()"

#     clusters = np.random.choice(np.arange(0,20,1), size=num_outlier_components, replace=True)
#     mixture_ratios = np.random.uniform(0,1,size=num_outlier_components) # Random mixture ratios make it highly unlikely that outliers resemble original clusters or each other
#     roll_probabilities = np.concatenate((np.ones(100), np.ones(280)*9, np.ones(100))) # Less likely (but not impossible) to only roll a small amount with a minimum likely offset of 5 hours
#     C_t = np.zeros(480)
#     for cluster, ratio in zip(clusters, mixture_ratios):
#         C_temp = C_func(cluster)
#         C_temp = np.roll(C_temp, -np.random.choice(np.arange(0,480,1), size=100000, p = roll_probabilities/roll_probabilities.sum()))
#         C_t = C_t + ratio*C_temp

#     C_t = C_t - np.min(C_t)
#     return C_t/np.max(C_t)
