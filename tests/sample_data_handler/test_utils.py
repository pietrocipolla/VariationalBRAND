from unittest import TestCase

from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.utils import get_labels_cluster_kmeans, get_training_set_example


class Test(TestCase):
    def test_get_labels_cluster_kmeans(self):
        X = generate_some_data_example()
        num_clusters = 5
        labels = get_labels_cluster_kmeans(X, num_clusters)
        #print(labels)
        #print(labels.shape)
        self.assertEqual(labels.shape, (500,))

    def test_get_training_set(self):
        X = generate_some_data_example()
        #print(X.shape)

        Y_learning, num_classes_learning = get_training_set_example(X)
        #print(Y_learning.shape, num_classes_learning)
        self.assertEqual(Y_learning.shape, (300, 2))
        self.assertEqual(num_classes_learning, 3)
