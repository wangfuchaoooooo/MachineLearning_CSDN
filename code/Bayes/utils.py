import numpy as np


def create_data():
    '''
    创建多个文档及其对应的标签
    :return: 文档内容，文档对应的标签
    '''
    docs_context = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    docs_labels = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return docs_context, docs_labels


def create_vocab_list(dataset):
    """
    创建词汇表
    :param dataset: 输入数据
    :return: 输入数据中不重复的词汇
    """
    vocab_set = []
    for doc in dataset:
        vocab_set = np.append(vocab_set, np.unique(doc))
    return np.unique(vocab_set)


def set_of_words_2_vec(vocabList, inputSet):
    '''
    将输入转换成词向量的形式
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return: 输入文档对应的词向量形式
    '''
    re_vec = np.zeros((len(vocabList, )))
    for word in inputSet:
        if word in vocabList:
            re_vec[np.argwhere(np.array(vocabList) == word)] = 1
        else:
            print('the word: %s is not in Vocabulary' % word)
    return re_vec

#
# if __name__ == "__main__":
#     docs_context, docs_labels = create_data()
#     vocabs = create_vocab_list(docs_context)
#     print('多文档生成的词汇表：', vocabs)
#     vec = set_of_words_2_vec(vocabs, docs_context[3])
#     print('输入文档对应的词向量：', vec)
#     # 多文档生成的词汇表： ['I' 'ate' 'buying' 'cute' 'dalmation' 'dog' 'flea' 'food' 'garbage' 'has'
#     #             'help' 'him' 'how' 'is' 'licks' 'love' 'maybe' 'mr' 'my' 'not' 'park'
#     #             'please' 'posting' 'problems' 'quit' 'so' 'steak' 'stop' 'stupid' 'take'
#     #             'to' 'worthless']
#     # 输入文档对应的词向量： [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
#     #              0. 0. 0. 1. 1. 0. 0. 1.]  # ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']
