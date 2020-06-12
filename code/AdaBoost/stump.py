import numpy as np


def stump_classify(x, dim, thresh_val, thresh_ineq):
    '''
    通过阈值比较对数据进行分类。
    所有在阈值一边的数据会分到类别-1，而另一边的数据会被分类到+1。
    通过数组过滤实现。
    '''
    retArray = np.ones((x.shape[0], 1))
    if thresh_ineq == 'lt':
        retArray[x[:, dim] <= thresh_val] = -1
    else:
        retArray[x[:, dim] > thresh_val] = -1
    return retArray


def build_stump(x, y, D):
    '''

    :param x: 输入数据
    :param y: 输入数据的标签
    :param D: 权重向量
    :return:
    '''
    x_mat = np.matrix(x)  # 转换成矩阵形式
    y_mat = np.matrix(y).T
    m, n = x.shape
    num_steps = 10
    best_stump = {}
    best_class_est = np.matrix(np.zeros((m, 1)))
    min_err = np.inf
    for i in range(n):  # 遍历所有特征
        row_min = x_mat[:, i].min()
        row_max = x_mat[:, i].max()
        step_size = (row_max - row_min) / num_steps
        for j in range(-1, num_steps + 1):
            for eq in ['lt', 'gt']:
                thresh_val = row_min + j * step_size
                pre_val = stump_classify(x, i, thresh_val, eq)
                err_arr = np.matrix(np.ones((m, 1)))
                err_arr[pre_val == y_mat] = 0
                weight_err = D.T * err_arr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, "
                      "the weighted error is %.3f"
                      % (i, thresh_val, eq, weight_err))
                if weight_err < min_err:
                    min_err = weight_err
                    best_class_est = pre_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = eq

        return best_stump, min_err, best_class_est
