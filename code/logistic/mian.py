import datetime
from utils import *

if __name__ == '__main__':
    X,y = read_data('./data/testSet.txt')
    time_s = datetime.datetime.now()
    w,b = stoc_grad_ascent(X,y,epochs=500)
    t = datetime.datetime.now()-time_s
    print(t)
    plot_best_fit(X,y,w,b)