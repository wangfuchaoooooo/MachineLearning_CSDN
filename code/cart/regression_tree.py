import numpy as np
from utils import *


def reg_leaf(dataset):
    '''定义回归树的叶子(该叶子上各样本标签的均值)'''
    return np.mean(dataset[:, -1])


def reg_err(dataset):
    '''定义连续数据的混乱度(总方差，即连续数据的混乱度=(该组各数据-该组数据均值)**2，即方差*样本数)'''
    return np.var(dataset[:-1]) * dataset.shape[0]

#判断是否是一棵树(字典)
def isTree(obj):
    return (type(obj).__name__=='dict')

#得到树所有叶节点的均值
def getMean(tree):
    #若子树仍然是树，则递归调用getMeant直到叶节点
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    return (tree['left']+tree['right'])/2.0

"""剪枝函数：对训练好的回归树，自上而下找到叶节点，用测试集来判断将这些叶节点合并是否能降低测试误差，若能则合并"""
def prune(tree,testData):
    #若无测试数据，则直接返回树所有叶节点的均值(塌陷处理)
    if testData.shape[0]==0:
        return getMean(tree)
    #若存在任意子集是树，则将测试集按当前树的最佳切分特征和特征值切分(子集剪枝用)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet,rSet=bin_split_data_set(testData,tree['spFeat'],tree['spVal'])
    #若存在任意子集是树，则该子集递归调用剪枝过程(利用刚才切分好的训练集)
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    #若当前子集都是叶节点，则计算该二叶节点合并前后的误差，决定是否合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=bin_split_data_set(testData,tree['spFeat'],tree['spVal'])
        errNotMerge=sum(np.power(lSet[:,-1].T.tolist()[0]-tree['left'],2))+sum(np.power(rSet[:,-1].T.tolist()[0]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errMerge=sum(np.power(testData[:,-1].T.tolist()[0]-treeMean,2))
        if errMerge<errNotMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree
