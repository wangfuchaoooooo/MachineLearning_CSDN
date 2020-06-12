from load_data import read_data
from fpgrowth import create_tree
from cond_pattern_base import find_prefix_path, mine_tree

if __name__ == '__main__':
    data = read_data()
    fp_tree, header_table = create_tree(data,3)
    fp_tree.display()
    condPats = find_prefix_path('x',header_table)
    print(condPats)
    condPats = find_prefix_path('z',header_table)
    print(condPats)
    condPats = find_prefix_path('r',header_table)
    print(condPats)
    mine_tree(header_table,3,set([]),[])