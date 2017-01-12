import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def multi_class_logistic_regression_ex3_ng():
    """Run multi-class logistic regression on test dataset.
        Example from Andrew Ng's coursera course
    """
    # ==================
    # load data

    # read matlab matrix file
    dataset = loadmat('data/ex3data1.mat')
    print(dataset.keys())

    y = dataset['y']  # 5000 x 1
    print('dims y: ', y.shape)

    X = dataset['X']  # 5000 x 400
    print('dims X: ', X.shape)

    # train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    num_samples_test = X_test.shape[0]

    # pick 20 examples and visualize them
    samples = np.random.choice(num_samples_test, 10)
    print('samples:', samples)
    plt.imshow(X_test[samples, :].reshape(-1, 20).T)
    plt.axis('off')

    # add constant for intercept - not needed for scikit learn
    # X = np.c_[np.ones((num_samples, 1)), X]
    # print('dims X: ', X.shape)

    # y = y.ravel() # turn column vector into 1d array

    # ==================
    # multi-class logistic regression
    regr = linear_model.LogisticRegression(C=10, penalty='l2', solver='liblinear', multi_class='ovr')
    regr.fit(X_train, y_train.ravel())

    pred = regr.predict(X_test)
    print('Training set accuracy: {} %'.format(np.mean(pred == y_test.ravel())*100))

    print('predicted class:', y_test[samples, :].ravel())
    plt.show()

multi_class_logistic_regression_ex3_ng()
