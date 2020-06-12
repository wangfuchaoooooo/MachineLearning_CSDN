import numpy as np
import numpy.linalg as nlinalg


def ridge_regress(x_mat, y_mat, lam=1.0):

    xTx = x_mat.T * x_mat
    denom = xTx + np.eye(x_mat.shape[1])*lam
    if nlinalg.det(denom) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (x_mat.T*y_mat)
    return ws


def test_ridge_regress(x, y):
    x_mat = np.matrix(x)
    y_mat = np.matrix(y).T
    # 数据标准化
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat-y_mean
    x_means = np.mean(y_mat,0)
    x_vars = np.var(y_mat,0)
    x_mat = (x_mat-x_means)/x_vars

    nums = 30
    w_mat = np.zeros((nums, x_mat.shape[1]))
    for i in range(nums):
        ws = ridge_regress(x_mat, y_mat, np.exp(i-10))
        w_mat[i,:] = ws.T
    return w_mat