import numpy as np
import collections


def kNN(new_data, train_data, labels, k=1):
    """
    :param new_data:  新数据
    :param train_data: 训练数据
    :param labels:  训练数据的标签
    :param k:  k值大小
    :return:  新数据的所属类别
    """

    train_set_rows = train_data.shape[0]  # 计算训练数据的数据量
    subs = train_data - np.tile(new_data, (train_set_rows, 1))
    distances = np.sqrt(np.sum(np.square(subs), axis=1))  # 计算新数据与训练集每个样本之间的欧式距离

    assert distances.shape != labels.shape, '维度不一致'
    sorted_distances = np.argsort(distances)  # 对距离进行排序

    k_labels = []
    for i in range(k):
        k_labels.append(labels[sorted_distances[i]][0])  # 找出排名前k的样本类别

    new_label = collections.Counter(k_labels).most_common(1)  # 对前k个样本类别进行统计，并输出最多的类别
    return new_label[0][0]
