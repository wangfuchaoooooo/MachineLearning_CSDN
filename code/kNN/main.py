import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from load_data import *
from knn import kNN
from plot import plot


if __name__ == "__main__":
    train_set = load_data(10)
    plot(train_set)
    new_data = ['ZL', 169, 2]

    train_data = np.array(train_set[['打斗镜头', '接吻镜头']])
    train_labels = np.array(train_set[['电影类别']])

    time_s = datetime.datetime.now()
    # ===========================手动实现======================
    label = kNN(new_data[1:], train_data, train_labels, k=3)
    time_e = datetime.datetime.now() - time_s
    print('用时：', time_e)
    print('新数据的类别：', label)
    # ===========================sklearn实现======================
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_data, train_labels)
    label = clf.predict([new_data[1:]])  # 输入是2D数据
    time_e = datetime.datetime.now() - time_s
    print('用时：', time_e)
    print('新数据的类别：', label[0])
