import numpy as np
import numpy.linalg as nlinalg


def lwlr_(test_data, x, y, k=1.0):
    x_mat = np.matrix(x)
    y_mat = np.matrix(y).T
    n = x.shape[0]
    weights = np.matrix(np.eye((n)))
    for j in range(n):
        diff_mat = test_data - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2 * k ** 2))
    xTx = x_mat.T * (weights * x_mat)
    if nlinalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_data*ws


def test_lwlr(testArr, x, y, k=1.):
    m = testArr.shape[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr_(testArr[i], x, y, k)
    return y_hat
