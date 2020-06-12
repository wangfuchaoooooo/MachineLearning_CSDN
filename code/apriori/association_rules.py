from apriori import apriori_gen


def calc_conf(freq_set, H, support_data, brl, min_conf=0.7):
    """
    计算规则可信度及找到满足最小可信度要求的规则
    :param freq_set: 频繁项集
    :param H: apriori_gen生成的频繁项集
    :param support_data: 频繁项集的支持度
    :param brl: big_rule_list
    :param min_conf: 最小可信度阈值
    :return: 满足最小可信度要求规则的列表
    """
    prunedH = []  # 初始化空规则列表
    for conseq in H:
        conf = support_data[freq_set] / support_data[freq_set - conseq]  # 计算可信度
        if conf >= min_conf:
            print(freq_set - conseq, ' --> ', conseq, ' conf: ', conf)
            brl.append((freq_set - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rules_from_conseq(freq_set, H, support_data, brl, min_conf=0.7):
    """
    从最初的的项集中生成更多的关联规则
    :param freq_set: 频繁项集
    :param H: 出现在规则右部的元素列表
    :param support_data: 频繁项集的支持度
    :param brl: big_rule_list
    :param min_conf: 最小可信度阈值
    :return:
    """
    m = len(H[0])
    if len(freq_set) > (m + 1):
        Hmp1 = apriori_gen(H, m + 1)
        Hmp1 = calc_conf(freq_set, Hmp1, support_data, brl, min_conf)
        if len(Hmp1) > 1:
            rules_from_conseq(freq_set, Hmp1, support_data, brl, min_conf)


def gen_rules(L, support_data, min_conf=0.7):
    """
    :param L: 频繁项集列表
    :param support_data: 包含频繁项集支持数据的列表
    :param min_conf: 最小可信度阈值
    :return: 可信度的规则列表
    """
    big_rule_list = []
    for i in range(1, len(L)):  # 无法从单个元素项集中构建关联规则，因此跳过第一个单元素项集
        for freq_set in L[i]:
            H1 = [frozenset([item]) for item in freq_set]  # 例子{0,1,2} ---> [{0},{1},{2}]
            if i > 1:  # 如果项集中的元素个数大于2个，作合并处理，通过rules_from_conseq方法完成
                rules_from_conseq(freq_set, H1, support_data, big_rule_list, min_conf)
            else:  # 如果项集中的元素个数只有2个，使用calc_conf方法，计算可信度
                calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list

# data = read_data()
# ret, support = apriori(data,0.5)
# rules = gen_rules(ret,support,0.7)
# min_conf=0.7 时的输出结果

# frozenset({1})  -->  frozenset({3})  conf:  1.0
# frozenset({5})  -->  frozenset({2})  conf:  1.0
# frozenset({2})  -->  frozenset({5})  conf:  1.0
# frozenset({5})  -->  frozenset({2, 3})  conf:  2.0
# frozenset({3})  -->  frozenset({2, 5})  conf:  2.0
# frozenset({2})  -->  frozenset({3, 5})  conf:  2.0
# [(frozenset({1}), frozenset({3}), 1.0), (frozenset({5}), frozenset({2}), 1.0),
# (frozenset({2}), frozenset({5}), 1.0), (frozenset({5}), frozenset({2, 3}), 2.0),
# (frozenset({3}), frozenset({2, 5}), 2.0), (frozenset({2}), frozenset({3, 5}), 2.0)]
