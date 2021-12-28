import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt


def get_labels_cluster_kmeans(X, num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(num_clusters, random_state=0)
    labels = kmeans.fit(X).predict(X)
    # print(kmeans.cluster_centers_)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    # plt.show()
    return kmeans.cluster_centers_

# data = loadtxt('/home/eb/PycharmProjects/VariationalBRAND/root/not_small_brand.csv', delimiter=',')
# Y = data
# T = 10
# get_labels_cluster_kmeans(Y, T)

def test_mu_var_DP_init_kmeans(Y,T):
    mu_var_DP = get_labels_cluster_kmeans(Y, T)
    return mu_var_DP


#test_mu_var_DP_init()