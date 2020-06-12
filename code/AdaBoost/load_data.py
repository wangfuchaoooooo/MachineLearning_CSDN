import numpy as np
import matplotlib.pyplot as plt


def read_data():
    data = np.array([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]])
    label = [1, 1, -1, -1, 1]
    return data,label


if __name__ =="__main__":
    data,label = read_data()
    x =data[:,0]
    y =data[:,1]
    # print(x)
    plt.scatter(x,y,c=label)
    plt.show()
