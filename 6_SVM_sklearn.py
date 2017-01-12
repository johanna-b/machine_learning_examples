import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat


def svm_ex6_ng():
    """ Run support vector machines.
        Example from Andrew Ng's coursera course
    """

    # =====================
    # load data

    # dataset = loadmat('data/ex6data1.mat')
    # dataset = loadmat('data/ex6data2.mat')
    dataset = loadmat('data/ex6data3.mat')
    print(dataset.keys())

    X = dataset['X']
    y = dataset['y']
    print('X:', X.shape, X[0, :])

    # =====================
    # init plotting
    gs = gridspec.GridSpec(2, 2)
    cur_img_index = 0
    fig = plt.figure(figsize=(10, 8), facecolor='white')

    # =====================
    # plot image

    pos = (y == 1).ravel()  # returns 1D array
    neg = (y == 0).ravel()

    fig.add_subplot(gs[cur_img_index])
    cur_img_index += 1
    plt.scatter(X[pos, 0], X[pos, 1],  color='red', marker='x', label='pos')
    plt.scatter(X[neg, 0], X[neg, 1],  color='black', marker='o', label='neg')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

    # =====================
    # linear SVM

    # 1.
    # svm = SVC(C=1.0, kernel='linear')

    # 2. gaussian kernel function (rbf)
    sigma = 0.1
    g = 1/(2*(sigma**2))
    svm = SVC(C=1.0, kernel='rbf', gamma=g)

    svm.fit(X, y.ravel())

    # =====================
    # plot SVM
    fig.add_subplot(gs[cur_img_index])
    cur_img_index += 1

    plot_svc(svm, X, y)

    plt.show()


def plot_svc(svc, X, y, h=0.01, pad=0.25):
    """Plot SVC"""

    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    pos = (y == 1).ravel()  # returns 1D array
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1],  color='red', marker='x', label='pos')
    plt.scatter(X[neg, 0], X[neg, 1],  color='black', marker='o', label='neg')

    # Support vectors indicated in plot by vertical lines
    # sv = svc.support_vectors_
    # plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)


svm_ex6_ng()
