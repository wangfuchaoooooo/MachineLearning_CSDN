import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def read_data(path):
    data = np.loadtxt(path)
    return data


# if __name__ == '__main__':
#     data = read_data('./data/reg_tree.txt')
#     plt.scatter(data[:,0], data[:,1])
#     plt.show()
