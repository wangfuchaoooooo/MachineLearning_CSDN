import numpy as np
from decision_tree import shannon_entropy,split_data,\
    choose_best_feature,DecisionTree,DecisionTreePredict
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot

def create_data():
    data = [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]]
    labels = ['yes', 'yes', 'no', 'no', 'no']
    categories = ['no surfacing','flippers']
    return np.array(data), np.array(labels), categories


if __name__ == "__main__":
    mydata, labels, categories = create_data()
    # print(mydata)
    # print(labels)

    # =============测试香农熵===============
    # re = shannon_entropy(ope, labels)
    # print(re)
    # ======================================

    # =============测试划分数据集===============
    # re_data, re_label = split_data(mydata, labels, 0,1)
    # print(re_data)
    # print(re_label)
    # ======================================

    # =============测试选择最好特征===============
    # bestFeat = choose_best_feature(mydata, labels)
    # print(bestFeat)
    # ======================================

    # =============测试===============
    feature_names = categories[:]
    mytree = DecisionTree(mydata, labels, categories)
    print(mytree)
    pre = DecisionTreePredict(mytree, [1,0], feature_names)
    print(pre)
    # ======================================

    # =============测试sklearn===============
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(mydata, labels)
    pre_label = tree.predict([[1,0]])
    print(pre_label)
    # export_graphviz(tree, out_file="./ope/tree.dot", feature_names=categories)
    # # 展示可视化图
    # (graph,) = pydot.graph_from_dot_file('./ope/tree.dot')
    # print(graph)
    # graph.write_png('./ope/tree.png')
    # ======================================