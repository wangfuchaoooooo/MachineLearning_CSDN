import numpy as np


def load_data(path):
    data = np.loadtxt(path)
    x_train = data[:, :-1]
    y_train = data[:, -1]
    return x_train, y_train


def sel_j_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    '''
    创建数据结构保存重要值
    '''

    def __init__(self, x_train, y_train, C, toler):
        self.X = x_train
        self.y = y_train
        self.C = C
        self.tol = toler
        self.m = x_train.shape[0]
        self.alphas = np.matrix(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.matrix(np.zeros((self.m, 2)))  # 第一列是eCache是否有效的标志位，第二列是实际值


def calEk(os, k):
    fxk = np.multiply(os.alphas, os.y).T * (os.X * os.X[k, :].T) + os.b
    Ek = fxk - os.y[k]
    return Ek


def sel_J(i, os, Ei):
    '''选择第二个alpha的值'''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcachelist = np.nonzero(os.eCache[:, 0].A)[0]
    if len(validEcachelist) > 1:
        for k in validEcachelist:
            if k == i:
                continue
            Ek = calEk(os, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = sel_j_rand(i, os.m)
        Ej = calEk(os, j)
    return j, Ej


def updateEk(os, k):
    Ek = calEk(os, k)
    os.eCache[k] = [1, Ek]


def calcWs(x_train,y_train,alphas):
    x_mat = np.matrix(x_train)
    y_mat = np.matrix(y_train).transpose()
    m,n = x_mat.shape
    w = np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*y_mat[i],x_mat[i,:].T)
    return w