import numpy as np


# 二分数据
def bin_split_data_set(dataset, feat, val):
    mat0 = dataset[np.nonzero(dataset[:, feat] > val)[0], :]  # 数组过滤选择特征大于指定值的数据
    mat1 = dataset[np.nonzero(dataset[:, feat] <= val)[0], :]  # 数组过滤选择特征小于指定值的数据
    return mat0, mat1


def choose_best_split(dataset, leafType=None, errType=None, ops=(1, 4)):
    """最佳特征以及最佳特征值选择函数
    leafType为叶节点取值，默认为None，可取regleaf，modelLeaf
    errType为数据误差(混乱度)计算方式，默认为None，可取regErr，modelErr
    ops[0]为以最佳特征及特征值切分数据前后，数据混乱度的变化阈值，若小于该阈值，不切分
    ops[1]为切分后两块数据的最少样本数，若少于该值，不切分
    回归树形状对ops[0],ops[1]很敏感，若这两个值过小，回归树会很臃肿，过拟合
    """
    tol_err = ops[0]
    tol_n = ops[1]
    m, n = dataset.shape
    err = errType(dataset)
    best_err = np.inf
    best_index = 0
    best_val = 0
    if len(set(dataset[:, -1].T)) == 1:  # 若只有一个类别
        return None, leafType(dataset)
    for featIndex in range(n - 1):
        for splitVal in set(dataset[:, featIndex].T):
            mat0, mat1 = bin_split_data_set(dataset, featIndex, splitVal)
            # 若切分后两块数据的最少样本数少于设定值，不切分
            if (mat0.shape[0] < tol_n) or (mat1.shape[0] < tol_n):
                continue
            new_err = errType(mat0) + errType(mat1)
            if new_err < best_err:
                best_index = featIndex
                best_val = splitVal
                best_err = new_err
    # 若以最佳特征及特征值切分后的数据混乱度与原数据混乱度差值小于阈值，不切分
    if (err - best_err) < tol_err:
        return None, leafType(dataset)
    mat0, mat1 = bin_split_data_set(dataset, best_index, best_val)
    # 若以最佳特征及特征值切分后两块数据的最少样本数少于设定值，不切分
    if (mat0.shape[0] < tol_n) or (mat1.shape[0] < tol_n):
        return None, leafType(dataset)
    return best_index, best_val


def create_tree(dataset, leafType=None, errType=None, ops=(1, 4)):
    """构建回归树"""
    feat, val = choose_best_split(dataset, leafType, errType, ops)
    if feat == None:
        return val
    reg_tree = {}
    reg_tree['spFeat'] = feat
    reg_tree['spVal'] = val
    lSet, rSet = bin_split_data_set(dataset, feat, val)
    reg_tree['left'] = create_tree(lSet, leafType, errType, ops)
    reg_tree['right'] = create_tree(rSet, leafType, errType, ops)
    return reg_tree
