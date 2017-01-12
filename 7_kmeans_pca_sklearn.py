import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn import cluster, decomposition
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def kmeans_ex7_ng():
    """Run kmeans.
        Example from Andrew Ng's coursera course
    """

    # =====================
    # load data
    dataset = loadmat('data/ex7data2.mat')
    print(dataset.keys())

    X = dataset['X']
    print('X:', X.shape, X[0, :])

    # =====================
    # kmeans
    num_clusters = 3
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=None)
    kmeans.fit(X)

    # =====================
    # plotting clusters

    plt.scatter(X[:, 0], X[:, 1], s=40, c=kmeans.labels_, cmap=plt.cm.prism)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='+', s=100, c='k', linewidth=2)

    # colors = ('red', 'black', 'blue')
    # markers = ('x', 'o', '+')
    # for i in range(num_clusters):
    #    clus = (kmeans.labels_ == i)
    #    plt.scatter(X[clus, 0], X[clus, 1],  color=colors[i], marker=markers[i], label='cluster_'+str(i))

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

    plt.show()


def kmeans_image_compression_ex7_ng():
    """use kmeans for compressing image"""

    # =====================
    # load data
    img = plt.imread('data/bird_small.png')
    img_shape = img.shape
    print('img:', img_shape)

    # reshape
    A = img / 255
    AA = A.reshape(img_shape[0]*img_shape[1], img_shape[2])
    print('AA:', AA.shape)

    # =====================
    # kmeans
    num_clusters = 16
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=None)
    kmeans.fit(AA)

    # smaller representation:
    # kmeans.labels_ (stores idx)
    # or kmeans.fit_predict(X)

    # =====================
    # recover
    AA_recovered = kmeans.cluster_centers_[kmeans.labels_]
    A_recovered = AA_recovered.reshape(img_shape[0], img_shape[1], img_shape[2])
    img_recovered = A_recovered * 255

    # =====================
    # display images
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    fig1 = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig1.set_title('Original')

    fig2 = fig.add_subplot(1, 2, 2)
    plt.imshow(img_recovered)
    fig2.set_title('Recovered')

    plt.show()


def pca_ex7_ng():
    """Run PCA"""

    # =====================
    # load data
    dataset = loadmat('data/ex7data1.mat')
    print(dataset.keys())

    X = dataset['X']
    print('X:', X.shape, X[0, :])

    # scatterplot of data
    fig = plt.figure(figsize=(9, 9), facecolor='white')
    fig.add_subplot(221)
    plt.scatter(X[:, 0], X[:, 1], s=40, c='k', cmap=plt.cm.prism)

    # =====================
    # standardizing the data
    scaler = StandardScaler()
    scaler.fit(X)
    mu = scaler.mean_
    X_scaled = scaler.transform(X)
    print('X scaled:', X_scaled.shape, X_scaled[0, :])

    # =====================
    # pca
    pca = decomposition.PCA()
    pca.fit(X_scaled)

    print('pca components (U):', pca.components_)
    print('mean:', mu)

    v = pca.explained_variance_
    print('variance ratio:', v)

    # =====================
    # plotting

    # p1 = mu[0] + pca.components_[0][0] * v[0]
    # p2 = mu[1] + pca.components_[0][1] * v[0]
    # plt.plot([mu[0], p1], [mu[1], p2], 'r-', lw=2)

    # two principal components as vectors
    plt.quiver(mu[0], mu[1], pca.components_[0][0] * v[0] * 1.5, pca.components_[0][1] * v[0] * 1.5, angles='xy', scale_units='xy', scale=1, color='r')
    plt.quiver(mu[0], mu[1], pca.components_[1][0] * v[1] * 1.5, pca.components_[1][1] * v[1] * 1.5, angles='xy', scale_units='xy', scale=1, color='r')

    # =====================
    # run pca with 1 component

    fig.add_subplot(222)

    pca2 = decomposition.PCA(n_components=1)
    pca2.fit(X_scaled)

    transformed = pca2.transform(X_scaled)
    print('transformed:', transformed.shape, transformed[0])

    recovered = pca2.inverse_transform(transformed)
    print('recovered:', recovered.shape, recovered[0])

    # =====================
    # display scatterplot of data and projected data (onto first principal component)

    # scaled original X
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=40, c='k', cmap=plt.cm.prism)
    # transformed
    plt.scatter(recovered[:, 0], recovered[:, 1], c='r', marker='x')

    plt.show()


def pca_faces_ex7_ng():
    """Use PCA for image compression"""

    # =====================
    # load data
    dataset = loadmat('data/ex7faces.mat')
    print(dataset.keys())

    # 5000 faces of 32x32
    X = dataset['X']
    print('X:', X.shape, X[0, :])

    # =====================
    # standardizing the data
    scaler = StandardScaler()
    scaler.fit(X)
    mu = scaler.mean_
    X_scaled = scaler.transform(X)
    print('X scaled:', X_scaled.shape, X_scaled[0, :])

    # =====================
    # pca, compression, recovery
    pca = decomposition.PCA(n_components=100)
    pca.fit(X_scaled)
    X_transformed = pca.transform(X_scaled)
    print('X transformed:', X_transformed.shape, X_transformed[0, :])

    X_recovered = pca.inverse_transform(X_transformed)
    print('X recovered:', X_recovered.shape, X_recovered[0, :])

    # =====================
    # display
    samples = np.linspace(0, 4500, num=10, dtype=np.int32)
    print('samples', samples)

    fig = plt.figure(facecolor='white')

    if True:  # display face images

        # original
        fig.add_subplot(311)
        plt.imshow(X[samples, :].reshape(-1, 32).T, cmap="gray")
        plt.axis('off')
        plt.title('original')

        # scaled and inverse scaled
        fig.add_subplot(312)
        X_scaled = scaler.inverse_transform(X_scaled)
        plt.imshow(X_scaled[samples, :].reshape(-1, 32).T, cmap="gray")
        plt.axis('off')
        plt.title('scaled and inverse scaled')

        # compressed
        fig.add_subplot(313)
        X_recovered = scaler.inverse_transform(X_recovered)
        plt.imshow(X_recovered[samples, :].reshape(-1, 32).T, cmap="gray")
        plt.axis('off')
        plt.title('PCA recovered (100 principal components)')

    else:   # display principal components

        print('pca components (U):', pca.components_.shape, pca.components_[0])

        gs = gridspec.GridSpec(6, 1)

        for i in range(6):
            fig.add_subplot(gs[i])
            indices = np.linspace(i*6, i*6+5, 6, dtype=np.int32)
            print(indices)
            imgs = pca.components_[indices, :]
            plt.imshow(imgs.reshape(-1, 32).T, cmap="gray")
            plt.axis('off')

    plt.show()


kmeans_ex7_ng()
kmeans_image_compression_ex7_ng()
pca_ex7_ng()
pca_faces_ex7_ng()
