import numpy as np
from utils import *


def k_means(data,k,dist_means = dist_eclud, create_cent=rand_cent):
    m = data.shape[0]
    cluster_assment = np.matrix(np.zeros((m,2)))
    centroids = create_cent(data,k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_j_i = dist_means(centroids[j,:], data[i,:])
                if dist_j_i<min_dist:
                    min_dist = dist_j_i
                    min_index = j
            if cluster_assment[i,0] != min_index:
                cluster_changed=True
            cluster_assment[i,:] = min_index,min_dist**2
        for cent in range(k):
            ptsInClust = data[np.nonzero(cluster_assment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids,cluster_assment