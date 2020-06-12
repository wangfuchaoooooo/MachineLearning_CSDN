from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

def creat_data_no_linear_2d(n):
    np.random.seed(1)
    x_11=np.random.randint(0,100,(n,1))
    x_12=10+np.random.randint(-5,5,(n,1))
    x_21=np.random.randint(0,100,(n,1))
    x_22=20+np.random.randint(0,10,(n,1))

    x_31=np.random.randint(0,100,(int(n/10),1))
    x_32=20+np.random.randint(0,10,(int(n/10),1))

    new_x_11=x_11*np.sqrt(2)/2-x_12*np.sqrt(2)/2
    new_x_12=x_11*np.sqrt(2)/2+x_12*np.sqrt(2)/2
    new_x_21=x_21*np.sqrt(2)/2-x_22*np.sqrt(2)/2
    new_x_22=x_21*np.sqrt(2)/2+x_22*np.sqrt(2)/2
    new_x_31=x_31*np.sqrt(2)/2-x_32*np.sqrt(2)/2
    new_x_32=x_31*np.sqrt(2)/2+x_32*np.sqrt(2)/2

    plus_samples=np.hstack([new_x_11,new_x_12,np.ones((n,1))])
    minus_samples=np.hstack([new_x_21,new_x_22,-np.ones((n,1))])
    err_samples=np.hstack([new_x_31,new_x_32,np.ones((int(n/10),1))])
    samples=np.vstack([plus_samples,minus_samples,err_samples])
    np.random.shuffle(samples)
    return samples


def plot_samples_2d(ax,samples):
    Y=samples[:,-1]
    position_p=Y==1
    position_m=Y==-1
    ax.scatter(samples[position_p,0],samples[position_p,1],
        marker='+',label='+',color='b')
    ax.scatter(samples[position_m,0],samples[position_m,1],
        marker='^',label='-',color='y')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
data=creat_data_no_linear_2d(100)
plot_samples_2d(ax,data)
ax.legend(loc='best')
plt.show()