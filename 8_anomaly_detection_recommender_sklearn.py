import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score
from scipy.optimize import minimize


def anomaly_detection_ex8_ng():
    """Run anomaly detection.
        Example from Andrew Ng's coursera course
    """

    # =====================
    # load data

    dataset = loadmat('data/ex8data1.mat')
    # dataset = loadmat('data/ex8data2.mat')
    print(dataset.keys())

    X = dataset['X']
    print('X:', X.shape, X[0, :])  # 307x2

    Xval = dataset['Xval']
    print('X_val:', Xval.shape, Xval[0, :])  # 307x2
    yval = dataset['yval']
    print('y_val:', yval.shape, yval[0, :])  # 307x1

    # =====================
    # display
    fig = plt.figure(facecolor='white')
    fig1 = fig.add_subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='k')
    plt.title("Outlier detection")
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

    # =====================
    # detecting outliers in a Gaussian distributed dataset.
    clf = EllipticEnvelope()
    clf.fit(X)

    # Calculate the decision function and use threshold to determine outliers
    y_pred = clf.decision_function(X).ravel()
    # print('y pred', y_pred)

    # =====================
    # find best threshold for outlier detection
    if False:
        samples = np.linspace(0.1, 10.0, num=100)
        best_f1 = 0.0
        best_perc = 0.0
        for sample in samples:
            Xval_pred = clf.decision_function(Xval)
            perc = sample
            th = np.percentile(Xval_pred, perc)
            outl = Xval_pred < th
            f1score = f1_score(yval, outl)
            print('f1 score (', sample, '):', f1score)

            if best_f1 < f1score:
                best_f1 = f1score
                best_perc = perc
        print('best f1:', best_f1, ', best perc:', best_perc)

    # set threshold for outlier detection
    percentile = 1.9  # 5.1 # 1.9 #best_perc # 1.9607843
    threshold = np.percentile(y_pred, percentile)
    outliers = y_pred < threshold
    # print('outliers:', X[outliers])

    # =====================
    # plot contours

    fig.add_subplot(2, 2, 2)

    # create the grid for plotting
    if False:
        xx, yy = np.meshgrid(np.linspace(0, 25, 200), np.linspace(0, 30, 200))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='blue', linestyles='dotted')

        threshold = np.percentile(y_pred, 1.0)
        plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='blue', linestyles='dotted')
        threshold = np.percentile(y_pred, 0.5)
        plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='blue', linestyles='dotted')

    # plot outliers
    plt.scatter(X[:, 0], X[:, 1], c='k')
    plt.scatter(X[outliers, 0], X[outliers, 1], c='r')
    print('num outliers:', sum(outliers))

    # samples_idx = yval == 1
    # print(yval[samples_idx])
    # print('X_val:', Xval.shape, Xval[0, :])  # 307x2
    # print(Xval[samples_idx])

    plt.show()


def recommender_system_ex8_ng():
    """Recommender system for movies"""

    # =====================
    # load data

    dataset = loadmat('data/ex8_movies.mat')
    # dataset = loadmat('data/ex8data2.mat')
    print('keys', dataset.keys())

    Y = dataset['Y']
    print('Y:', Y.shape, Y[0, 0])  # 1682, 943 rating 0-5

    R = dataset['R']
    print('R:', R.shape, R[0, 0])

    print('mean:', Y[1, R[1, :]].mean())

    # debug output
    if False:
        users = 4
        movies = 5
        features = 3

        params_data = loadmat('data/ex8_movieParams.mat')
        X = params_data['X']
        Theta = params_data['Theta']

        X_sub = X[:movies, :features]
        Theta_sub = Theta[:users, :features]
        Y_sub = Y[:movies, :users]
        R_sub = R[:movies, :users]

        params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

        # c = cost(params, Y_sub, R_sub, features)
        # print('cost', c)

        J, grad = cost(params, Y_sub, R_sub, features, 1.5)
        print('cost: j, grad:', J, grad)

    # =====================
    #  read movie info

    # movie_idx = {}
    # f = open('data/movie_ids.txt')
    # for line in f:
    #    tokens = line.split(' ')
    #    tokens[-1] = tokens[-1][:-1]
    #    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

    # =====================
    # add ratings

    ratings = np.zeros((1682, 1))

    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5

    Y = np.append(Y, ratings, axis=1)
    R = np.append(R, ratings != 0, axis=1)

    # =====================
    # random init data

    movies = Y.shape[0]
    users = Y.shape[1]
    features = 10
    learning_rate = 10.

    X = np.random.random(size=(movies, features))
    Theta = np.random.random(size=(users, features))
    params = np.concatenate((np.ravel(X), np.ravel(Theta)))

    Ymean = np.zeros((movies, 1))
    Ynorm = np.zeros((movies, users))

    for i in range(movies):
        idx = np.where(R[i, :] == 1)[0]
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    # =====================
    # recommender system
    fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),
                    method='CG', jac=True, options={'maxiter': 100})
    print(fmin)

    X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
    Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

    print('X:', X.shape, Theta.shape)

    # =====================
    # predict

    predictions = X * Theta.T
    my_preds = predictions[:, -1] + Ymean
    sorted_preds = np.sort(my_preds, axis=0)[::-1]
    print(sorted_preds[:10])

    # TODO: not supported in sklearn


def cost(params, Y, R, num_features, learning_rate):
    """calculate cost for recommender system"""

    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)

    # =====================
    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    # =====================
    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # =====================
    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad


anomaly_detection_ex8_ng()
recommender_system_ex8_ng()
