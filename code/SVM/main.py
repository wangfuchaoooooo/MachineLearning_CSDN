from utils import *
from simple_SMO import simple_smo
from SMO import smop

if __name__ == '__main__':
    data_path = './ope/testSet.txt'
    x_train, y_train = load_data(data_path)
    # b, alphas = simple_smo(x_train,y_train,0.6,0.001,40)
    b, alphas = smop(x_train, y_train, 0.6, 0.001, 40)
    w = calcWs(x_train, y_train, alphas)

    # test
    x_mat = np.matrix(x_train)
    pre = x_mat[2] * np.matrix(w) + b
    print('预测值：', pre)
    print('实际值：', y_train[2])
