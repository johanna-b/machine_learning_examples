import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import learning_curve
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures


def regularized_linear_regression_ex5_ng():
    """Run regularized linear regression.
        Example from Andrew Ng's coursera course
    """

    # =====================
    # load data
    dataset = loadmat('data/ex5data1.mat')
    print(dataset.keys())

    X_train = dataset['X']      # 12 x 1
    X_cv = dataset['Xval']      # 21 x 1
    X_test = dataset['Xtest']   # 21 x 1
    print('dims X_train: ', X_train.shape)
    print('dims X_cv: ', X_cv.shape)
    print('dims X_test: ', X_test.shape)

    y_train = dataset['y']      # 12 x 1
    y_cv = dataset['yval']      # 21 x 1
    y_test = dataset['ytest']   # 21 x 1
    print('dims y_train: ', y_train.shape)
    print('dims y_cv: ', y_cv.shape)
    print('dims y_test: ', y_test.shape)

    # =====================
    # plot data

    fig = plt.figure(figsize=(8, 10), facecolor='white')

    fig.add_subplot(321)
    plt.scatter(X_train[:, 0], y_train,  color='black', label='data')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of dam')
    plt.legend(loc="best")
    plt.title("lin reg with 1 feature")

    # =====================
    # linear regression

    regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    regr.fit(X_train, y_train)
    plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)

    # =====================
    # combining test and cv (sklearn does that for us)

    X = np.concatenate((X_train, X_cv))
    y = np.concatenate((y_train, y_cv))
    print('dims X: ', X.shape)

    # =====================
    # learning curve

    fig.add_subplot(322)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes = np.linspace(.1, 1.0, 20)    # start, stop, num samples

    train_sizes, train_scores, test_scores = learning_curve(
        regr, X, y, cv=3, train_sizes=train_sizes)  # does k-fold cross validation
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    # =====================
    # add features 1

    # add features
    for i in range(2, 8):
        X_train = np.c_[X_train, X_train[:, 0]**i]
        X_cv = np.c_[X_cv, X_cv[:, 0]**i]

    X = np.concatenate((X_train, X_cv))
    y = np.concatenate((y_train, y_cv))
    print('dims X: ', X.shape)

    # plot training set
    fig.add_subplot(323)
    plt.scatter(X_train[:, 0], y_train,  color='red', marker='x', label='data')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of dam')
    plt.title("lin reg with 7 features (manual)")

    # linear regression
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    regr.fit(X_train, y_train)

    # plot range for x
    plot_x = np.linspace(-60, 45).reshape(-1, 1)
    for i in range(2, 8):
        plot_x = np.c_[plot_x, plot_x[:, 0]**i]

    # using coefficients to calculate y
    plot_y = regr.predict(plot_x)
    plt.plot(plot_x[:, 0], plot_y, label='Scikit-learn Linear Regression')
    plt.xlim(-60, 60)

    # =====================
    # add features 2

    poly = PolynomialFeatures(degree=8)
    X_train_poly = poly.fit_transform(X_train[:, 0].reshape(-1, 1))

    regr2 = linear_model.Ridge(alpha=20)
    regr2.fit(X_train_poly, y_train)

    fig.add_subplot(325)
    plt.scatter(X_train[:, 0], y_train,  color='red', marker='x', label='data')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of dam')
    plt.title("lin reg with 7 features (manual)")

    # plot range for x
    plot_x = np.linspace(-60, 45)
    # using coefficients to calculate y
    plot_y = regr2.intercept_ + np.sum(regr2.coef_ * poly.fit_transform(plot_x.reshape(-1, 1)), axis=1)
    plt.plot(plot_x, plot_y, label='Scikit-learn Ridge (alpha={})'.format(regr2.alpha))
    plt.xlim(-60, 60)
    plt.ylim(-80, 80)
    plt.title("scikit-learn ridge with Polynomial features")

    # =====================
    # learning curve 1

    fig.add_subplot(324)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes = np.linspace(.1, 1.0, 20)    # start, stop, num samples

    train_sizes, train_scores, test_scores = learning_curve(
        regr, X, y, cv=3, train_sizes=train_sizes)  # does k-fold cross validation
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.ylim(-0.2, 1)

    plt.legend(loc="best")

    # =====================
    # learning curve 2

    fig.add_subplot(326)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes = np.linspace(.1, 1.0, 20)    # start, stop, num samples

    train_sizes, train_scores, test_scores = learning_curve(
        regr2, X, y, cv=3, train_sizes=train_sizes)  # does k-fold cross validation
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.ylim(-0.2, 1)
    plt.legend(loc="best")

    # =====================

    plt.tight_layout()
    plt.show()


regularized_linear_regression_ex5_ng()
