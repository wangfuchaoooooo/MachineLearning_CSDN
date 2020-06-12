import numpy as np


def dist_eclud(va, vb):
    '''计算两个向量之间的欧氏距离'''
    return np.sqrt(np.sum(np.power(va - vb, 2)))


def rand_cent(data, k):
    '''随机构建k个随机质心'''
    n = data.shape[1]
    centroids = np.matrix(np.zeros((k, n)))
    for i in range(n):
        min_i = np.min(data[:, i])
        range_i = np.max(data[:, i]) - min_i
        centroids[:, i] = min_i + range_i * np.random.rand(k, 1)
    return centroids
