from root.controller.sample_data_handler.utils import get_labels_cluster_kmeans
from jax import numpy as jnp


def calculate_robust_parameters(X, num_classes):
    labels = get_labels_cluster_kmeans(X, num_classes)
    list_inv_cov_mat = []
    list_robust_mean = []
    i = 0
    while i< num_classes:
        X_cluster_i = X[jnp.where(labels == i)]
        temp_inv_cov_mat, temp_robust_mean = mcd(X_cluster_i)
        list_inv_cov_mat.append(jnp.array(temp_inv_cov_mat))
        list_robust_mean.append(jnp.array(temp_robust_mean))
        i+=1
    return  list_robust_mean, list_inv_cov_mat

def calculate_robust_parameters_labels(X, num_classes, labels):
    list_inv_cov_mat = []
    list_robust_mean = []
    i = 1
    while i< num_classes+1:
        X_cluster_i = X[jnp.where(labels == i)]
        temp_inv_cov_mat, temp_robust_mean = mcd(X_cluster_i)
        list_inv_cov_mat.append(jnp.array(temp_inv_cov_mat))
        list_robust_mean.append(jnp.array(temp_robust_mean))
        i+=1
    return  list_robust_mean, list_inv_cov_mat

def mcd(X):
    # Load libraries
    import scipy as sp
    from sklearn.covariance import MinCovDet

    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    return inv_covmat, robust_mean
