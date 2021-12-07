#https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
#Load libraries
import numpy as np

from controller.sample_data_hanlder.data_generator import generate_sample_data


def calculate_robust_parameters(X, labels, num_classes):
    list_inv_cov_mat = []
    list_robust_mean = []
    i = 0
    while i< num_classes:
        X_cluster_i = X[np.where(labels == i)]
        temp_inv_cov_mat, temp_robust_mean = mcd(X_cluster_i)
        list_inv_cov_mat.append(temp_inv_cov_mat)
        list_robust_mean.append(temp_robust_mean)
        i+=1
    return list_inv_cov_mat, list_robust_mean

#todo settare altri paraemtri robusti

def mcd(X):
    # Load libraries
    import scipy as sp
    from sklearn.covariance import MinCovDet

    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    return inv_covmat, robust_mean


# X, labels, num_classes = generate_sample_data(5,500)
# print(calculate_robust_parameters(X, labels, num_classes))