import numpy as np
from stump import *


def adaboost_trian(x, y, num_it=40):
    week_class_arr = []
    m = x.shape[0]
    D = np.matrix(np.ones((m, 1)) / m)  # 初始化权重向量D
    agg_class_est = np.matrix(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, err, class_est = build_stump(x, y, D)
        print('D: ', D.T)
        alpha = 0.5 * np.log((1 - err) // max(err, 1e-16))  # 计算alpha值
        best_stump['alpha'] = alpha
        week_class_arr.append(best_stump)
        print('class_est: ', class_est.T)
        expon = np.multiply(np.matrix(-1 * alpha * y).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)
        agg_class_est += np.multiply(alpha, class_est)
        print('agg_class_est: ', agg_class_est.T)
        agg_err = np.multiply(np.sign(agg_class_est) != np.matrix(y).T, np.ones((m, 1)))
        err_rate = np.sum(agg_err) / m
        print('total error: ', err_rate, '\n')
        if err_rate == 0:
            break
    return week_class_arr


def adaClassify(test, classifierArr):
    dataMatrix = np.matrix(test)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = dataMatrix.shape[0]
    aggClassEst = np.matrix(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stump_classify(dataMatrix, classifierArr[i]['dim'],
                                  classifierArr[i]['thresh'],
                                  classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += np.multiply(classifierArr[i]['alpha'], classEst)
        print(aggClassEst)
    return np.sign(aggClassEst)
