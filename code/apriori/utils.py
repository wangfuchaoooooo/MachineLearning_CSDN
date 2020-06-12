def create_C1(data):
    """创建所有候选项集的集合"""
    C1 = []
    for tran in data:
        for item in tran:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)

# Test
# 输入数据 data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]；create_C1(data)
# 输出：frozenset({1}) frozenset({2}) frozenset({3}) frozenset({4}) frozenset({5})


def scan_D(data,candidate_set,min_support=0.5):
    '''
    :param data: 包含候选集合的数据集
    :param candidate_set: 候选集合
    :param min_support: 最小支持度
    :return: 满足最小支持的候选集，每个候选集的支持度
    '''

    C1 = [i for i in candidate_set]
    cs_cnt = {}
    for tid in data:  # 循环数据集中的每条交易记录tid
        for cs in C1: # 循环每个候选集cs
            if cs.issubset(tid):  # 检查cs是否是tid的子集
                if cs not in cs_cnt:
                    cs_cnt[cs]=1
                else:cs_cnt[cs]+=1  # 如果是，则cs的计数增加
    num_items = len(data)
    ret_list = []
    support_data = {}
    for k in cs_cnt:  # 对每个候选集
        support = cs_cnt[k]/num_items  # 计算其支持度
        if support>=min_support:  # 如果其支持度不小于最小值，则保留该候选集
            ret_list.append(k)
        support_data[k] = support
    return ret_list,support_data

# Test
# 输入数据 data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# 输入数据 candidate_set = [frozenset({1}) frozenset({2}) frozenset({3}) frozenset({4}) frozenset({5})]
# 输入数据 min_support = 0.5

# 输出数据：
# ret_list：[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
# support_data：{frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}