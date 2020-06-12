import numpy as np
from model_tree import *


def reg_tree_eval(model, inDat):
    return float(model)


def model_tree_eval(model, inDat):
    n = inDat.shape[1]
    X = np.matrix(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def tree_fore_cast(tree, inData, modelEval=reg_tree_eval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spFeat']] > tree['spVal']:
        if isTree(tree['left']):
            return tree_fore_cast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return tree_fore_cast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def create_fore_cast(tree, testData, modelEval=reg_tree_eval):
    m = len(testData)
    yHat = np.matrix(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = tree_fore_cast(tree, np.matrix(testData[i]), modelEval)
    return yHat