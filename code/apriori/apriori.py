from utils import create_C1, scan_D


def apriori_gen(Lk, k):
    ret_list = []
    len_LK = len(Lk)
    for i in range(len_LK):
        for j in range(i + 1, len_LK):
            L1 = list(Lk[i])[:k - 2].sort()
            L2 = list(Lk[j])[:k - 2].sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])
    return ret_list


def apriori(data, min_support=0.5):
    C1 = create_C1(data)
    L1, support_data = scan_D(data, C1, min_support)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0: # 集合中项的个数大于0时
        Ck = apriori_gen(L[k - 2], k) # 构建一个k个项构成的候选集列表
        Lk, supk = scan_D(data, Ck, min_support) # 确认每个候选集都是频繁的
        support_data.update(supk)
        L.append(Lk) # 保留频繁项
        k += 1 # 为构建k+1项组成的候选集列表做准备
    return L, support_data
# Test
# 输入数据 data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# 输入数据 min_support = 0.5
# 输出：[[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})], [frozenset({1, 3}),
#      frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5})], [frozenset({2, 3, 5})], []]
