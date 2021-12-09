import numpy as np

def get_training_set(X):
    # example get data from 3 of the 5 clusters
    num_classes_learning = 3
    labels = get_labels_cluster_kmeans(X, num_classes_learning)
    Y_learning = X[np.where(labels < num_classes_learning)]

    return Y_learning, num_classes_learning


def get_labels_cluster_kmeans(X, num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(num_clusters, random_state=0)
    labels = kmeans.fit(X).predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    # plt.show()
    return labels

