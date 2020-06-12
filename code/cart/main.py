import numpy as np
import matplotlib.pyplot as plt
from load_data import read_data
from regression_tree import *
from model_tree import *
from predict import *

if __name__ == '__main__':
    # data = read_data('./data/reg_tree.txt')
    # data = read_data('./data/ex2.txt')
    # reg_tree = create_tree(dataset=data,leafType=reg_leaf,errType=reg_err) # 够建回归树
    # print('回归树：',reg_tree)
    # test_data = read_data('./data/ex2test.txt')
    # test_mat = np.matrix(test_data)
    # p_reg_tree = prune(reg_tree,test_mat)
    # print('回归树剪枝后：', p_reg_tree)

    # data = read_data('./data/exp2.txt')
    # test_data = read_data('./data/expTest.txt')
    # test_mat = np.matrix(test_data)
    # model_tree = create_tree(dataset=data,leafType=model_leaf,errType=model_err) # 够建模型树
    # print('模型树：', model_tree)
    # p_model_tree = model_prune(model_tree,data,test_mat)
    # print('模型树剪枝后：', p_model_tree)

    train = read_data('./data/bikeSpeedVsIq_train.txt')
    test = read_data('./data/bikeSpeedVsIq_test.txt')
    reg_tree = create_tree(dataset=train, leafType=reg_leaf, errType=reg_err)
    y_hat = create_fore_cast(reg_tree,test[:,0])
    corr = np.corrcoef(y_hat,test[:,1],rowvar=0)
    print(corr)
