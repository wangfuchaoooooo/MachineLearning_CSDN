import matplotlib.pyplot as plt
import numpy as np


def plot(train_set):
    movies_name = np.array(train_set[['电影名称']])
    train_data = np.array(train_set[['打斗镜头', '接吻镜头']])
    train_labels = np.array(train_set[['电影类别']])
    X = train_data[:, 0]
    y = train_data[:, 1]
    plt.scatter(X, y)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    for i in range(len(X)):
        plt.annotate(movies_name[i][0], xy=(X[i], y[i]), xytext=(X[i] + 0.1, y[i] + 0.1))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标

    plt.show()
