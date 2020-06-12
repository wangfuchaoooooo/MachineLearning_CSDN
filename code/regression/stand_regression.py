import numpy as np
import numpy.linalg as nlinalg


def stand_reg(x,y):
    x_mat = np.matrix(x)
    y_mat = np.matrix(y).T
    xTx = x_mat.T*x_mat
    if nlinalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I*(x_mat.T*y_mat)
    return ws
