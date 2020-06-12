from st_tree import treeNode


def create_tree(data, min_support=1): # 创建 FP树
    header_table = {} # 初始化头指针列表
    for trans in data:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data[trans]  # 第一次扫描数据集，统计所有项的频度
    for k in list(header_table.keys()): # 删除频度小于给定值的项
        if header_table[k] < min_support:
            del header_table[k]
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0: return None, None # 都不满足最小支持度的要求，直接结束程序返回
    for k in header_table:
        header_table[k] = [header_table[k], None] # 调整头指针列表的结构，使其可以链接到其他元素，即变成一个链表形式
    ret_tree = treeNode('Null tree', 1, None) # 创建 FP的树根
    for tran_set, count in data.items():  # 第二次遍历数据集，
        local_D = {}
        for item in tran_set:
            if item in freq_item_set: # 只考虑那些频繁项
                local_D[item] = header_table[item][0]
        if len(local_D) > 0:
            ordered_item = [v[0] for v in sorted(local_D.items(), key=lambda p: p[1], reverse=True)] # 排序
            update_tree(ordered_item, ret_tree, header_table, count) # 更新树
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].increase(count)
    else:
        in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node2test, target_node):
    """确保节点连接指向树中该元素项的每个实例"""
    while node2test.node_link is not None:
        node2test = node2test.node_link
    node2test.node_link = target_node


# 测试结果
#  Null tree   1
#    z   5
#     r   1
#     x   3
#      t   3
#       s   2
#        y   2
#       r   1
#        y   1
#    x   1
#     s   1
#      r   1