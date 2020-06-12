import numpy as np
from utils import *

def linear_solve(dataset):
    '''叶节点计算方法：该叶节点所有样本的标准线性回归模型'''
    m, n = dataset.shape
    x_mat = np.matrix(np.ones((m, n)))
    x_mat[:, 1:n] = dataset[:, 0:n - 1]  # x_mat第一列为常数项1
    y_mat = dataset[:, -1]
    y_mat = np.matrix(y_mat).T
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0:
        print("矩阵为奇异矩阵，不可逆，尝试增大ops的第二个参数")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws, x_mat, y_mat


def model_leaf(dataset):
    ws, X, Y = linear_solve(dataset)
    return ws


def model_err(dataset):
    '''误差计算方法：用线性模型对数据拟合，计算真实值与拟合值之差，求差值的平方和'''
    ws, X, Y = linear_solve(dataset)
    y_hat = X * ws
    return sum(np.power(Y - y_hat, 2))


# 判断是否是一棵树(字典)
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 剪枝函数：对训练好的模型树，自上而下找到叶节点，用测试集来判断将这些叶节点合并是否能降低测试误差，若能则合并
def model_prune(tree, trainData, testData):
    m, n = testData.shape
    # 若无测试数据，则直接返回树所有叶节点的均值(塌陷处理)
    if m == 0:
        return tree
    # 若存在任意子集是树，则将测试集按当前树的最佳切分特征和特征值切分(子集剪枝用)
    # 同时将训练集也按当前树的最佳切分特征和特征值切分(子集剪枝用)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = bin_split_data_set(testData, tree['spFeat'], tree['spVal'])
        lTrain, rTrain = bin_split_data_set(trainData, tree['spFeat'], tree['spVal'])
    # 若存在任意子集是树，则该子集递归调用剪枝过程(利用刚才切分好的训练集)
    if isTree(tree['left']):
        tree['left'] = model_prune(tree['left'], lTrain, lSet)
    if isTree(tree['right']):
        tree['right'] = model_prune(tree['right'], rTrain, rSet)

    # 若当前子集都是叶节点，则计算该二叶节点合并前后的误差，决定是否合并
    """
    模型树，两个叶节点合并前的误差=((左叶子真实值-拟合值)的平方和+(右叶子真实值-拟合值)的平方和)
    模型树，两个叶节点合并后的误差=(左右真实值-左右拟合值)的平方和
    难点在于如何求左右拟合值，即求上层节点的回归系数wsMerge：用上层节点的traindata,通过linearSolve(traindata)求得
    上层节点的traindata在lTrain,rTrain的递归中已经求好了
    """
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = bin_split_data_set(testData, tree['spFeat'], tree['spVal'])
        lSetX = np.matrix(np.ones((lSet.shape[0], n)))
        rSetX = np.matrix(np.ones((rSet.shape[0], n)))
        lSetX[:, 1:n] = lSet[:, 0:n - 1]
        rSetX[:, 1:n] = rSet[:, 0:n - 1]

        errNotMerge = np.sum(np.power(np.array(lSet[:, -1].T) - lSetX * tree['left'], 2)) + np.sum(
            np.power(np.array(rSet[:, -1].T) - rSetX * tree['right'], 2))
        # 难点在于求上层节点的回归系数wsMerge：用上层节点的traindata,通过linearSolve(traindata)求得
        wsMerge = model_leaf(trainData)
        testDataX = np.matrix(np.ones((m, n)));
        testDataX[:, 1:n] = testData[:, 0:n - 1]
        errMerge = np.sum(np.power(np.array(testData[:, -1].T) - testDataX * wsMerge, 2))
        if errMerge < errNotMerge:
            print("merging")
            return wsMerge
        else:
            return tree
    else:
        return tree