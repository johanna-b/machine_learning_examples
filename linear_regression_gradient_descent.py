import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def warmup_excercise():
    return np.identity(5)


def compute_cost(X, y, theta):
    """compute linear regression cost function"""

    m = y.shape[0]
    h = np.dot(X, theta)  # matrix vector multiplication
    J = (1/(2*m)) * np.sum(np.square(h-y))

    return J


def gradient_descent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=3000):
    """Run gradient descent to find theta that minimizes cost function"""

    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))

    for idx in range(0, num_iters):
        h = np.dot(X, theta)
        theta = theta - ((alpha/m) * np.dot(X.transpose(), (h-y)))
        J_history[idx] = compute_cost(X, y, theta)

    return theta, J_history


def linear_regression():
    """Run linear regression on test dataset. Example from Andrew Ng's coursera course"""

    # =====================
    # load and prepare data
    dataset = pd.read_csv('data/ex1data1.txt', header=None)
    # print('data dims: ', dataset.shape)

    X = np.ones((dataset.shape[0], 1))  # first column, x_0, intercept
    X = np.append(X, dataset.values[:, 0:1], 1)  # x_1

    y = dataset.values[:, np.newaxis, 1]

    # =====================
    # plot data
    fig = plt.figure(figsize=(10, 8), facecolor='white')

    fig1 = fig.add_subplot(221)
    plt.scatter(X[:, 1], y,  color='black', label='data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.xlim(4, 24)  # viewport along x-axis

    # =====================
    # check cost function
    theta = np.zeros((2, 1))
    cost = compute_cost(X, y, theta)
    print('cost:', cost)

    # =====================
    # run gradient descent, to fit model
    iterations = 1500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    print('theta:', theta)

    fig2 = fig.add_subplot(222)
    plt.plot(J_history)
    plt.ylabel('Cost of J')
    plt.xlabel('Iterations')

    # =====================
    # plot linear regression line
    plt.subplot(221)
    plt.plot(X[:, 1], np.dot(X, theta), color='blue', linewidth=3, label='Linear Regression')
    plt.legend()

    # =====================
    # predict some examples
    predict1 = np.dot([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of', predict1*10000)

    predict2 = np.dot([1, 7], theta)
    print('For population = 70,000, we predict a profit of', predict2*10000)

    # =====================
    # draw contour and surface plot of cost for different thetas

    # compute cost for grid of theta 0 and theta 1
    theta0_vals = np.linspace(-10, 10, 50)  # 100 samples over -10 and 10
    theta1_vals = np.linspace(-1, 4, 50)
    xx, yy = np.meshgrid(theta0_vals, theta1_vals, indexing='xy')
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # calculate cost based on grid of coefficients
    for (i, j), elem in np.ndenumerate(J_vals):
        J_vals[i, j] = compute_cost(X, y, theta=[[xx[i, j]], [yy[i, j]]])

    # contour plot
    fig3 = fig.add_subplot(223)
    plt.contour(xx, yy, J_vals, np.logspace(-2, 3, 20), cmap=plt.get_cmap('plasma'), linewidths=3)  # show contour lines
    plt.scatter(theta[0], theta[1], c='r')  # show computed theta
    plt.ylabel('Theta 1')
    plt.xlabel('Theta 0')

    # surface plot
    fig4 = fig.add_subplot(224, projection='3d')
    fig4.plot_surface(X=xx, Y=yy, Z=J_vals, rstride=1, cstride=1, alpha=0.6, cmap=plt.get_cmap('plasma'))
    plt.ylabel(r'Theta 1 $\theta_1$')
    plt.xlabel(r'Theta 0 $\theta_0$')

    plt.show()


# print(warmup_excercise())
linear_regression()
