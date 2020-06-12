import numpy as np
from utils import *


def simple_smo(x_train, y_train, C, toler, maxIter):
    """

    :param x_train: 数据集
    :param y_train: 数据标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return: 常数b，alpha向量
    """
    x_mat = np.matrix(x_train)  # 转矩阵方便运算
    y_mat = np.matrix(y_train).transpose()  # 转矩阵并转置方便运算
    b = 0  # 初始化常数b
    m, n = x_mat.shape  # 数据集的大小
    alphas = np.matrix(np.zeros((m, 1)))  # 初始化alpha矩阵
    iter = 0  # 初始化循环值
    while iter < maxIter:  # 当前循环值小于最大循环值时进行循环
        alpha_pairs_changed = 0  # 记录两个alpha值是否改变
        for i in range(m):  # 遍历整个数据集
            fxi = np.multiply(alphas, y_mat).T * (x_mat * x_mat[i, :].T) + b  # 计算预测类别
            Ei = fxi - y_mat[i]  # 预测值与实际值之间的误差
            if ((y_mat[i] * Ei < -toler) and (alphas[i] < C)) \
                    or ((y_mat[i] * Ei > toler) and (alphas[i] > C)):
                '''误差判断条件，如果误差过大则对alpha进行优化,正负间隔都会被测试，并且保证alpha值不能等于0或C。
                后面alpha>C或alpha<0都会被调整到0或C，如果在条件上alpha已经等于0或C，alpha就不会再增大或减小，
                就不会对其进行优化了'''
                j = sel_j_rand(i, m)  # 随机挑选第二个alpha值
                fxj = np.multiply(alphas, y_mat).T * (x_mat * x_mat[j, :].T) + b  # 计算预测类别
                Ej = fxj - y_mat[j]  # 预测值与实际值之间的误差
                alpha_i_old = alphas[i].copy()  # 拷贝数值，方便后面进行新旧值之间的比较
                alpha_j_old = alphas[j].copy()
                if y_mat[i] != y_mat[j]:  # 将计算L和H，用于将alpha值调整到0和C之间，如果L和H相等，不进行任何操作
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2 * x_mat[i, :] * x_mat[j, :].T - x_mat[i, :] * x_mat[i, :].T - \
                      x_mat[j, :] * x_mat[j, :].T  # alpha[j],即第二个alpha，的最优修改量
                if eta >= 0:  # 退出当前循环
                    print('eta>=0')
                    continue
                alphas[j] -= y_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old) < 0.00001:  # 检测alpha[j]是否有轻微改变，改变的话退出循环
                    print('j not moving enough')
                    continue
                alphas[i] += y_mat[j] * y_mat[i] * (alpha_j_old - alphas[j])  # 改变alpha[i]
                b1 = b - Ei - y_mat[i] * (alphas[i] - alpha_i_old) * x_mat[i, :] * x_mat[i, :].T - \
                     y_mat[j] * (alphas[j] - alpha_j_old) * x_mat[i, :] * x_mat[j, :].T  # 改alpha[i]设置常数项
                b2 = b - Ej - y_mat[i] * (alphas[i] - alpha_i_old) * x_mat[i, :] * x_mat[j, :].T - \
                     y_mat[j] * (alphas[j] - alpha_j_old) * x_mat[j, :] * x_mat[j, :].T  # 改alpha[j]设置常数项
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha_pairs_changed += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas
