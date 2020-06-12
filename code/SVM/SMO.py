import numpy as np
from utils import *


def innerL(i, os):
    Ei = calEk(os, i)
    if ((os.y[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) \
            or ((os.y[i] * Ei > os.tol) and (os.alphas[i] > os.C)):
        j, Ej = sel_J(i, os, Ei)
        alpha_i_old = os.alphas[i].copy()  # 拷贝数值，方便后面进行新旧值之间的比较
        alpha_j_old = os.alphas[j].copy()
        if os.y[i] != os.y[j]:  # 将计算L和H，用于将alpha值调整到0和C之间，如果L和H相等，不进行任何操作
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - \
              os.X[j, :] * os.X[j, :].T
        if eta >= 0:  # 退出当前循环
            print('eta>=0')
            return 0
        os.alphas[j] -= os.y[j] * (Ei - Ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        updateEk(os, j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:  # 检测alpha[j]是否有轻微改变，改变的话退出循环
            print('j not moving enough')
            return 0
        os.alphas[i] += os.y[j] * os.y[i] * (alpha_j_old - os.alphas[j])  # 改变alpha[i]
        b1 = os.b - Ei - os.y[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[i, :].T - \
             os.y[j] * (os.alphas[j] - alpha_j_old) * os.X[i, :] * os.X[j, :].T  # 改alpha[i]设置常数项
        b2 = os.b - Ej - os.y[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[j, :].T - \
             os.y[j] * (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[j, :].T  # 改alpha[j]设置常数项
        if 0 < os.alphas[i] < os.C:
            os.b = b1
        elif 0 < os.alphas[j] < os.C:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smop(x_train, y_train, C, toler, maxIter, kTup=('lin', 0)):
    os = optStruct(np.matrix(x_train), np.matrix(y_train).transpose(), C, toler)
    iter = 0
    entiry_set = True
    alpha_pairs_changed = 0
    while iter < maxIter and (alpha_pairs_changed > 0 or entiry_set):
        alpha_pairs_changed = 0
        if entiry_set:
            for i in range(os.m):
                alpha_pairs_changed += innerL(i, os)
                print('fullSet, iter: %d i: %d, pairs changed %d' %(iter, i, alpha_pairs_changed))
            iter += 1
        else:
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < 0))[0]
            for i in nonBoundIs:
                alpha_pairs_changed += innerL(i, os)
                print('non-bound, iter: %d i: %d, pairs changed %d'%(iter, i, alpha_pairs_changed))
            iter += 1
        if entiry_set:
            entiry_set = False
        elif alpha_pairs_changed == 0:
            entiry_set = True
        print("iteration number: %d" % iter)
    return os.b, os.alphas
