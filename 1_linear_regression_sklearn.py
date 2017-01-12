import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
# from sklearn.model_selection import train_test_split


def linear_regression_test():
    """ perform linear regression on a test dataset.
        Dataset is split into training and testing dataset.
    """

    # ==================
    # load the diabetes dataset
    diabetes = datasets.load_diabetes()  # attributes: data (X), target (y)

    # ==================
    # look at dataset
    print('data dimensions: ', diabetes.data.shape)
    print('row 0: ', diabetes.data[0, :])

    # ==================
    # use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]  # all rows, 1 axis, with column index 2

    print('X dimensions: ', diabetes_X.shape)
    print('X row 0: ', diabetes_X[0, :])

    # ==================
    # split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]  # leave 20 samples out for testing
    diabetes_X_test = diabetes_X[-20:]

    print('X train dims: ', diabetes_X_train.shape)

    # split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # alternatively: diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, y)

    # ==================
    # scatterplot

    plt.scatter(diabetes_X_train, diabetes_y_train,  color='0.95')  # grey scale
    plt.scatter(diabetes_X_test, diabetes_y_test,  color=(0.2, 0.2, 0.2))  # color='black'

    # create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    # fit_intercept: use theta_zero; normalize: normalizes X

    # train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # attributes of trained model (with test data
    print('Coefficients: \n', regr.intercept_, regr.coef_)
    print("Cost: (Mean squared error): %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # plot line graph
    plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

    plt.show()


def linear_regression_ex1_ng():
    """Run linear regression on test dataset.
        Example from Andrew Ng's coursera course
    """

    # ==================
    # read data
    dataset = pandas.read_csv('data/ex1data1.txt', header=None)
    print('data: ', dataset.head(10))
    print('dims: ', dataset.shape)

    X_train = dataset.values[:, np.newaxis, 0]
    print('dims x_train: ', X_train.shape)
    print('x_train[0]: ', X_train[0])

    y_train = dataset.values[:, np.newaxis, 1]
    print('dims y_train: ', y_train.shape)
    print('y_train[0]: ', y_train[0])

    # ==================
    # plot training data
    print("Plotting data")
    plt.scatter(X_train, y_train,  color='black')

    # ==================
    # linear regression
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    regr.fit(X_train, y_train)

    print('Coefficients: \n', regr.intercept_, regr.coef_)

    # ==================
    # draw linear regression line
    plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)

    # ==================
    # test
    X_test = [3.5, 7.0]
    X_test = np.reshape(X_test, (-1, 1))

    y_test = regr.predict(X_test)
    print('predicted vals: ', y_test)

    # ==================
    # plot test data
    plt.scatter(X_test, y_test,  color='red')

    plt.show()


def linear_regression_ex1_multivariate_ng():
    """Run multivariate linear regression on test dataset.
        Example from Andrew Ng's coursera course
    """

    # ==================
    # read data
    dataset = pandas.read_csv('data/ex1data2.txt', header=None)
    print('data: ', dataset.head(10))
    print('dims: ', dataset.shape)

    X_train = dataset.values[:, 0:2]
    print('dims x_train: ', X_train.shape)
    print('x_train[0]: ', X_train[0])

    y_train = dataset.values[:, np.newaxis, 2]
    print('dims y_train: ', y_train.shape)
    print('y_train[0]: ', y_train[0])

    # ==================
    # linear regression
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    regr.fit(X_train, y_train)

    print('Coefficients: \n', regr.intercept_, regr.coef_)

    # ==================
    # test
    X_test = [1650, 3]
    print('test for house of (squ meters, bedrooms): ', X_test)
    X_test = np.reshape(X_test, (-1, 2))

    y_test = regr.predict(X_test)
    print('predicted vals: ', y_test[0][0])


linear_regression_test()
linear_regression_ex1_ng()
linear_regression_ex1_multivariate_ng()
