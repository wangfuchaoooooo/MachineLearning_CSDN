import numpy as np
import matplotlib.pyplot as plt
from load_data import read_data
from stand_regression import *
from lwlr import test_lwlr
from ridge_regression import test_ridge_regress
if __name__ == '__main__':
    # data, la = read_data('./data/ex0.txt')
    # ws = stand_reg(data,la)
    #
    # plt.scatter(data[:,1], la)
    # y_hat = np.matrix(data)*ws
    # print('corrcoef:', np.corrcoef(y_hat.T,np.matrix(la))) # 计算预测值与实际值之间的相关系数，确定结果好坏
    # plt.plot(data[:,1], y_hat, color='r')
    # plt.show()

    # lwlr
    # y_hat = test_lwlr(data, data, la)
    # plt.scatter(data[:, 1], la)
    # x_mat = np.matrix(data)
    # Ind = x_mat[:,1].argsort(0)
    # data_sort = x_mat[Ind][:,0,:]
    # plt.plot(data_sort[:, 1], y_hat[Ind], color='r')
    # plt.show()

    # ridge
    data, la = read_data('./data/abalone.txt')
    ridge_weights = test_ridge_regress(data,la)
    plt.plot(ridge_weights)
    plt.show()

