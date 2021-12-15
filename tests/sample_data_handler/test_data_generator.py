from unittest import TestCase
import matplotlib.pyplot as plt
from numpy import savetxt

from bin.save_load_numpy import load_data_nupy
from controller.sample_data_handler.utils import get_labels_cluster_kmeans
from model import variational_parameters
from root.controller.sample_data_handler.data_generator import generate_some_data_example


class Test(TestCase):
    def test_generate_some_data_example(self):
        X = generate_some_data_example()
        # print(X.shape)
        #print(X.shape[1])
        self.assertEqual(X.shape, (1000, 2))

        savetxt('data.csv', X, delimiter=',')

        X = load_data_nupy()
        num_clusters = 5
        labels = get_labels_cluster_kmeans(X, num_clusters)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
        # plt.scatter(variational_parameters.nIW_MIX_VAR.mu[:, 0], variational_parameters.nIW_MIX_VAR.mu[:, 1],
        #             color='red')
        # plt.scatter(variational_parameters.nIW_DP_VAR.mu[:, 0], variational_parameters.nIW_DP_VAR.mu[:, 1],
        #             color='blue')
        plt.show()
