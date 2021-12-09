import numpy as np
import matplotlib.pyplot as plt #do not remove
from jax import numpy as jnp

def generate_some_data():
    #example : 5, 2d gaussians
    num_clusters = 5
    num_samples = 500

    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=num_samples, centers=num_clusters,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    return jnp.array(X)


def get_training_set(X):
    # example get data from 3 of the 5 clusters
    num_classes_learning = 3
    labels = get_labels_cluster_kmeans(X, num_classes_learning)
    Y_learning = X[np.where(labels < num_classes_learning)]

    # labels_learning = labels[np.where(labels < num_classes_learning)]
    # plt.scatter(X_learning[:, 0], X_learning[:, 1], c=labels_learning, s=40, cmap='viridis');
    # plt.show()

    return Y_learning, num_classes_learning


def get_labels_cluster_kmeans(X, num_clusters):
    # Plot the data with K Means Labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(num_clusters, random_state=0)
    labels = kmeans.fit(X).predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    # plt.show()
    return labels


