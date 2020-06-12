from math import log
import numpy as np


def shannon_entropy(train_data, train_labels):
    """
    计算熵
    :param train_data: 需要计算想农熵的数据集
    :param train_labels: 数据集的标签
    :return: 香农熵
    """
    num_samples = len(train_data)  # 统计样本数量
    label_counts = {}  # 定义标签字典
    assert train_data.shape[0] == train_labels.shape[0], '维度不匹配'
    for sample, current_label in zip(train_data, train_labels):  # 统计每个标签的数量
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0  # 初始化想农熵
    for k in label_counts:  # 计算总的信息熵
        prob = float(label_counts[k]) / num_samples
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data(train_data, train_labels, feature, value):
    """
    划分数据集
    :param train_data: 待划分的数据集
    :param feature: 划分数据集的特征
    :param value: 特征的返回值
    :return: 分割后的数据集(即删除最好特征列，剩余的数据)
    """
    rest_data = []
    rest_labels = []
    for index, sampleVec in enumerate(train_data):
        if sampleVec[feature] == value:
            rest_labels.append(train_labels[index])
            temp = np.concatenate([sampleVec[:feature], sampleVec[feature + 1:]])
            rest_data.append(temp)
    return np.array(rest_data), np.array(rest_labels)


def choose_best_feature(train_data, train_labels):
    """
    选择最好特征
    :param train_data: 训练数据
    :param train_labels: 训练数据标签
    :return: 样本最好特征的序号
    """
    num_features = len(train_data[0])
    base_entropy = shannon_entropy(train_data, train_labels)
    best_info_gain = 0
    best_feature = -1
    for n in range(num_features):
        feat_values = [sample[n] for sample in train_data]
        unique_values = set(feat_values)
        new_entropy = 0
        for value in unique_values:
            sub_data, sub_labels = split_data(train_data, train_labels, n, value)
            prob = len(sub_data) / float(len(train_data))
            new_entropy += prob * shannon_entropy(sub_data, sub_labels)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = n
    return best_feature


def majority_vote(clist):
    """

    :param clist: 输入列表
    :return: 列表中最多的元素
    """
    import collections
    return collections.Counter(clist).most_common(1)[0][0]


def DecisionTree(train_data, train_labels, categories):
    """
    构建决策树
    :param train_data: 训练数据
    :param train_labels: 训练数据标签
    :param categories: 特征名称列表
    :return: 构建完成的决策树
    """
    class_list = train_labels
    if len(set(class_list)) == 1:
        return class_list[0]
    if len(train_data[0]) == 0:
        return majority_vote(class_list)
    best_feat = choose_best_feature(train_data, train_labels)
    best_feat_label = categories[best_feat]
    my_tree = {best_feat_label: {}}
    del (categories[best_feat])
    feat_values = [sample[best_feat] for sample in train_data]
    unique_feat_values = set(feat_values)
    for value in unique_feat_values:
        subcategories = categories[:]
        sub_data, sublabels = split_data(train_data, train_labels, best_feat, value)
        my_tree[best_feat_label][value] = DecisionTree(sub_data, sublabels, subcategories)
    return my_tree


def DecisionTreePredict(tree, test_data, feature_names):
    """

    :param tree: 构建完成的决策树
    :param test_data: 测试数据
    :param feature_names: 特征名称列表
    :return: 预测值
    """
    first_feat = list(tree.keys())[0]
    second_dict = tree[first_feat]
    feat_index = feature_names.index(first_feat)
    for k in second_dict.keys():
        if test_data[feat_index] == k:
            if type(second_dict[k]) is dict:
                class_label = DecisionTreePredict(second_dict[k], test_data, feature_names)
            else:
                class_label = second_dict[k]
    return class_label
