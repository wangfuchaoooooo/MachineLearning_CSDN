import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from k_means import k_means
from bin_k_means import biKmeans

if __name__ == '__main__':
    # data = read_data('./data/testSet.txt')
    data = read_data('./data/testSet2.txt')
    plt.scatter(data[:,0], data[:,1])
    # cent, clus = k_means(data,4)
    # plt.scatter(cent[:,0].T.tolist(), cent[:,1].T.tolist())
    cent, clus = biKmeans(data,4)
    cent_x = []
    cent_y = []
    for c in cent:
        cent_x.append(np.array(c)[0][0])
        cent_y.append(np.array(c)[0][1])
    plt.scatter(cent_x, cent_y)
    plt.show()


