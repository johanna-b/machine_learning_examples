
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def logistic_regression_ex2_ng():
    """Run (unregularized) logistic regression on test dataset.
        Example from Andrew Ng's coursera course
    """

    # ==================
    #  read data
    dataset = pandas.read_csv('data/ex2data1.txt', header=None)
    # print('data: ', dataset.head(10))
    print('data dims: ', dataset.shape)

    X_train = dataset.values[:, 0:2]
    print('dims x_train: ', X_train.shape)
    print('x_train[0]: ', X_train[0])

    y_train = dataset.values[:, 2]
    print('dims y_train: ', y_train.shape)
    print('y_train[0]: ', y_train[0])
    print('\n')

    # ==================
    #  plot data
    print("Plotting data")
    pos = np.where(y_train == 1)
    neg = np.where(y_train == 0)
    plt.scatter(X_train[pos, 0], X_train[pos, 1], marker='x', color='red', label='admitted')
    plt.scatter(X_train[neg, 0], X_train[neg, 1], marker='o', color='green', label='not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

    # ==================
    #  logistic regression
    regr = linear_model.LogisticRegression(C=1e5)  # C: inverse regularization strength!
    # (in this example we do not want regularization, so we choose a high number)
    regr.fit(X_train, y_train)

    print('\n\nCoefficients:', regr.intercept_, regr.coef_)
    # should be -24.9, 0.2, 0.2

    # ==================
    #  plot decision boundary
    min_x = np.amin(X_train, axis=0)
    max_x = np.amax(X_train, axis=0)
    xx, yy = np.mgrid[min_x[0]:max_x[0]:1, min_x[1]:max_x[1]:1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = regr.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, probabilities, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    # ==================
    #  test

    # 1 example
    testdata = [45, 85]
    X_test = np.reshape(testdata, (-1, 2))
    #print(X_test.shape, X_test)
    prob = regr.predict_proba(X_test)
    print('For a student with scores 45 and 85, we predict an admission probability of',
          round(prob[0, 1]*100))
    # should be 77

    # test accuracy
    y_test = regr.predict(X_train)
    accuracy = (np.sum(y_test == y_train) / y_test.shape[0]) * 100
    print('Accuracy:', accuracy)
    # should be 89

    plt.show()


def map_feature(X1, X2):
    """Creates combination of parameters X1 and X2: X1, X2, X1*X2, X1^2 * X2,..."""

    degree = 6
    out = np.ones((X1.shape[0], 1))

    for i in range(1,degree+1):
        for j in range(0,i+1):
            new_col = (X1**(i-j)) * (X2**j)
            out = np.append(out, new_col, 1)

    return out


def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))


def logistic_regression_regularized_ex2_ng():
    """Run (regularized) logistic regression on test dataset.
        Example from Andrew Ng's coursera course
    """

    # ==================
    # read data
    dataset = pandas.read_csv('data/ex2data2.txt', header=None)
    # print('data: ', dataset.head(10))
    print('data dims: ', dataset.shape)

    X_train = dataset.values[:, 0:2]
    print('dims x_train: ', X_train.shape)
    print('x_train[0]: ', X_train[0])

    y_train = dataset.values[:, 2]
    print('dims y_train: ', y_train.shape)
    print('y_train[0]: ', y_train[0])
    print('\n')

    # ==================
    # plot data
    print("Plotting data")
    pos = np.where(y_train == 1)
    neg = np.where(y_train == 0)
    plt.scatter(X_train[pos, 0], X_train[pos, 1], marker='x', color='red', label='y=1')
    plt.scatter(X_train[neg, 0], X_train[neg, 1], marker='o', color='green', label='y=0')
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    plt.legend()

    X1 = X_train[:, np.newaxis, 0]
    X2 = X_train[:, np.newaxis, 1]
    # print('X1 shape:', X1.shape)
    # print('X2 shape:', X2.shape)
    X = map_feature(X1, X2)
    print('Shape of features', X.shape)

    # ==================
    # logistic regression
    regr = linear_model.LogisticRegression(C=1)  # C: inverse regularization strength!

    # find optimal parameters in grid search
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000] }
    # regr = grid_search.GridSearchCV(linear_model.LogisticRegression(penalty='l2',intercept_scaling=1, dual=False, fit_intercept=True), param_grid)

    regr.fit(X, y_train)
    print('\n\nCoefficients:', regr.intercept_, regr.coef_)
    # print('\n\nCoefficients:', regr.best_estimator_.intercept_, regr.best_estimator_.coef_)
    # print('best params:', regr.best_params_)

    # ==================
    # plot decision boundary
    xx, yy = np.mgrid[-1:1.5:0.05, -1:1.5:0.05]
    grid = np.c_[xx.ravel(), yy.ravel()]
    g1 = grid[:, np.newaxis, 0]
    g2 = grid[:, np.newaxis, 1]
    grid_X = map_feature(g1, g2)
    probs = regr.predict_proba(grid_X)[:, 1].reshape(xx.shape)
    print('probability:', probs.shape)
    plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    # ==================
    # test accuracy
    y_test = regr.predict(X)
    accuracy = (np.sum(y_test == y_train) / y_test.shape[0]) * 100
    print('Accuracy:', accuracy)
    # should be 83

    t1 = np.array([0.0]).reshape(1, 1)
    t2 = np.array([-0.5]).reshape(1, 1)
    t_X = map_feature(t1, t2)
    prob = regr.predict_proba(t_X)
    print('Probability for 0.0, -0.5:', round(prob[0, 1]*100))

    plt.show()

logistic_regression_ex2_ng()
logistic_regression_regularized_ex2_ng()
