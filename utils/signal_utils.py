import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import correlate


def get_cross_correlation(sample_1,sample_2):
    # Taking the first signal as reference
    
    cross_correlations=[]
    reference = []
    
    for i in range(5):  # Assuming 5 channels
        cross_corr = correlate(sample_1[:,i], sample_2[:,i])
        ref = correlate(sample_1[:,i], sample_1[:,i])
        cross_correlations.append(cross_corr) 
        reference.append(ref)
    cross_correlations= np.stack(cross_correlations)
    reference = np.stack(reference)
    
    return np.abs(cross_correlations.mean(axis=-1)-reference.mean(axis=-1))

def get_euclidean_distance(sample_1,sample_2):
    # (datapoints,channels)
    distance = sample_1- sample_2
    distance = np.linalg.norm(distance,axis=0)
    return distance

def get_cosine_similarity(sample_1,sample_2):
    # (datapoints,channels)    
    cosine_list = []
    for s1_dim,s2_dim in zip(sample_1.T,sample_2.T):
        cosine_list.append(cosine(s1_dim,s2_dim))
    return np.stack(cosine_list)