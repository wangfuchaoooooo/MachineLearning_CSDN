from fpgrowth import create_tree


def ascend_tree(leaf_node, prefixPath):  # ascends from leaf node to root
    if leaf_node.parent is not None:
        prefixPath.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefixPath)


def find_prefix_path(base_pattern, header_table):  # treeNode comes from header table
    tree_node = header_table[base_pattern][1]
    condPats = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            condPats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return condPats

# 测试结果
#  condPats = find_prefix_path('x',header_table)
#     {frozenset({'z'}): 3}
#  condPats = find_prefix_path('z',header_table)
#     {}
#  condPats = find_prefix_path('r',header_table)
#     {frozenset({'z'}): 1, frozenset({'x'}): 1, frozenset({'y', 'z', 'x'}): 1}


def mine_tree(header_table, min_support, pre_fix, freq_item_list):
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        new_freq_set = pre_fix.copy()
        new_freq_set.add(basePat)
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(basePat, header_table)
        myCondTree, myHead = create_tree(cond_patt_bases, min_support)
        if myHead is not None:
            print('conditional tree for: ',new_freq_set)
            myCondTree.display()
            mine_tree(myHead, min_support, new_freq_set, freq_item_list)

# 测试结果
# conditional tree for:  {'y'}
#   Null tree   1
#    z   3
#     x   3
# conditional tree for:  {'y', 'x'}
#   Null tree   1
#    z   3
# conditional tree for:  {'t'}
#   Null tree   1
#    y   3
#     z   3
#      x   3
# conditional tree for:  {'t', 'z'}
#   Null tree   1
#    y   3
# conditional tree for:  {'t', 'x'}
#   Null tree   1
#    y   3
#     z   3
# conditional tree for:  {'t', 'z', 'x'}
#   Null tree   1
#    y   3
# conditional tree for:  {'s'}
#   Null tree   1
#    x   3
# conditional tree for:  {'x'}
#   Null tree   1
#    z   3