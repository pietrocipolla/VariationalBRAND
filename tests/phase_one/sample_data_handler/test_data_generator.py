from unittest import TestCase

from controller.sample_data_handler.data_generator import generate_sample_data, get_training_set


class Test(TestCase):
    def test_generate_sample_data(self):
        n_samples = 500
        X, labels = generate_sample_data(5, n_samples)
        # print(X.shape)
        self.assertEqual(X.shape, (500, 2))


class Test(TestCase):
    def test_get_training_set(self):
        n_samples = 500
        X, labels = generate_sample_data(5, n_samples)
        # print(X.shape)
        num_classes_learning = 3
        X_learning, labels_learning, num_classes_learning = get_training_set(X, labels,num_classes_learning)
        #print(X_learning.shape)
        self.assertEqual(X_learning.shape, (300, 2))
