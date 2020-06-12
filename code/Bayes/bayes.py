import numpy as np
import collections

"""
Bayes算法的伪代码
1. 计算每个类别中的文档数目
2. 训练每篇文档：
3. for 文档类别：
4.     对每个类别：
5.        如果词条出现在文档中----->增加该词的计数值
6.        增加所有词条的计数值
7. 对每个类别：
8.     对每个词条：
9.         该词条的数目/总词条数目=条件概率
10. 返回每个类别的条件概率
"""


def cond_probability(train_matrix, train_category):
    '''
    计算条件概率, 此方法可以进行多分类
    :param train_matrix: 所有文档词向量组成的祖矩阵
    :param train_category: 文档类别
    :return: 词向量中每个元素的概率，每种类别的占比
    '''
    num_train_docs = len(train_matrix)  # 统计样本数量
    num_words = len(train_matrix[0])  # 统计词汇量的大小
    train_category_ = train_category[:]  # 复制 train_category
    num_category = len(np.unique(train_category_))  # 统计类别数量

    pAbusive = np.zeros((num_category, 1))
    for category, count in collections.Counter(train_category_).most_common():  # 计算每个类别的占比
        pAbusive[category] = count / num_train_docs

    p_num = np.ones((num_category, num_words))  # 初始化为1： 防止某一个概率为0，导致相乘结果为0
    p_demon = np.ones((num_category, 1)) * 2

    for i in range(num_train_docs):
        p_num[train_category[i]] += train_matrix[i]
        p_demon[train_category[i]] += np.sum(train_matrix[i])

    p_vec = np.log(p_num / p_demon) # 取对数：防止过多较小数相乘，导致结果数值下溢出

    del train_category_
    return p_vec, pAbusive


def classifyNB(test_data, p_vec, p_class):
    '''
    对输入文档进行预测
    :param test_data: 输入测试文档
    :param p_vec:  词向量中每个元素的概率
    :param p_class: 每种类别的占比
    :return:
    '''
    p = np.reshape(np.sum(test_data * p_vec, axis=1), (2, 1)) + np.log(p_class)
    return np.argmax(p)



