import numpy as np


def read_data(data_path):
    data = np.loadtxt(data_path)
    return data[:, :-1], data[:, -1]


def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


'''
1. 初始化回归系数w
2. for N次数：
3.     计算整个数据集的梯度gradient
4.     使用w = w + alpha*gradient更新回归系数
5.     返回回归系数
'''

def grad_ascent(x_train, y_train, learning_rate=0.001, epochs=500):
    x_mat = np.matrix(x_train)
    y_mat = np.matrix(y_train).transpose()
    rows, cols = x_mat.shape
    args = np.ones((cols + 1, 1))
    add_1 = np.ones((rows,))
    x_mat = np.column_stack((x_mat, add_1))
    for k in range(epochs):
        y_pre = sigmoid(x_mat * args)
        loss = y_mat - y_pre
        args = args + learning_rate * x_mat.transpose() * loss
    w = args[:-1]
    b = np.array(args[-1])
    return w, b


def stoc_grad_ascent(x_train, y_train, learning_rate=0.01, epochs=500):
    rows, cols = x_train.shape
    args = np.ones((cols + 1, ))
    add_1 = np.ones((rows,))
    x_train = np.column_stack((x_train, add_1))
    for j in range(epochs):
        data_index = np.arange(rows)
        for i in range(rows):
            learning_rate = 4/(j+i+1)+0.01
            # learning_rate = learning_rate*0.95
            rand_index = int(np.random.uniform(0,len(data_index)))
            y_pre = sigmoid(sum(x_train[rand_index] * args))
            loss = y_train[rand_index] - y_pre
            args = args + learning_rate * x_train[rand_index] * loss
            np.delete(data_index, rand_index)
    w = args[:-1]
    b = args[-1]
    return w, b

def plot_best_fit(x_train, y_train, weight, bias):
    import matplotlib.pyplot as plt
    rows = x_train.shape[0]
    x0 = []; x1 = []; y0 = []; y1 = []
    for i in range(rows):
        if y_train[i] == 0:
            x0.append(x_train[i, 0])
            y0.append(x_train[i, 1])
        else:
            x1.append(x_train[i, 0])
            y1.append(x_train[i, 1])
    ax = plt.subplot(111)
    ax.scatter(x0, y0, s=30, c='r')
    ax.scatter(x1, y1, s=30, c='g')
    x = np.arange(-3, 3, 0.1)
    y = (-bias - weight[0] * x) / weight[1]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     x1 = np.arange(-5, 5)
#     x2 = np.arange(-200, 200)
#     y1 = sigmoid(x1)
#     y2 = sigmoid(x2)
#     plt.subplot(211)
#     plt.plot(x1,y1)
#     plt.axvline(x=0, ls="--", c="red")  # 添加垂直直线
#     plt.xlabel('x')
#     plt.ylabel('sigmoid(x)')
#     plt.subplot(212)
#     plt.plot(x2,y2)
#     plt.axvline(x=0, ls="--", c="red")  # 添加垂直直线
#     plt.xlabel('x')
#     plt.ylabel('sigmoid(x)')
#     plt.show()
