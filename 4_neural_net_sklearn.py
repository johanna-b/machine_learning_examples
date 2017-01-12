import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import gridspec

# import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

def neural_net_ex4_ng():
    """Run a neural network on test dataset.
        Example from Andrew Ng's coursera course
    """
    # ==================
    # read data
    dataset = loadmat('data/ex4data1.mat')
    print(dataset.keys())

    y = dataset['y'] # 5000 x 1
    print('dims y: ', y.shape)
    # print('y[0]: ', y[0])

    X = dataset['X'] # 5000 x 400
    print('dims X: ', X.shape)
    # print('X[0]: ', X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    num_samples_test = X_test.shape[0]

    # ==================
    # display data

    # pick 20 examples and visualize them
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    fig.add_subplot(651)
    samples = np.random.choice(num_samples_test, 10)
    print('samples:', samples)
    plt.imshow(X_test[samples, :].reshape(-1, 20).T, cmap="Greys")
    plt.axis('off')

    # ==================
    # run neural net
    hidden_layer_size = 25

    mlp = MLPClassifier(hidden_layer_sizes=(25,), max_iter=20, alpha=1e-4,
                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp.fit(X_train, y_train.ravel())

    predictions = mlp.predict(X_test)
    print('Test set accuracy: {} %'.format(np.mean(predictions == y_test.ravel())*100))

    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
    print('coeffs shape', (mlp.coefs_[0]).shape)

    # ==================
    # display coefficients of hidden layer
    fig.add_subplot(652)
    plt.imshow(mlp.coefs_[0][:, 0].reshape(20, 20))
    plt.axis('off')

    gs = gridspec.GridSpec(6, 5)
    cur_img_idx = 5

    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, range(hidden_layer_size)):
        fig.add_subplot(gs[cur_img_idx])
        plt.imshow(coef.reshape(20, 20), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        plt.axis('off')
        cur_img_idx += 1

    plt.show()


neural_net_ex4_ng()
