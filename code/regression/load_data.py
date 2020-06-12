import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1]


# if __name__ == '__main__':
#     data, la = read_data('./data/ex0.txt')
#     plt.scatter(data[:,1], la)
#     plt.show()
