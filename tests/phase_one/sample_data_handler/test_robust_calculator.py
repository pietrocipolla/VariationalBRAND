from unittest import TestCase

from controller.sample_data_handler.data_generator import generate_sample_data, get_training_set
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters


class Test(TestCase):
    def test_mcd(self):
        n_samples = 500
        X, labels = generate_sample_data(5, n_samples)

        num_classes_learning = 3
        X_learning, labels_learning, num_classes_learning = get_training_set(X, labels, num_classes_learning)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(X_learning, labels_learning, num_classes_learning)

        #print(list_robust_mean)
        #print(list_inv_cov_mat[1].shape)

        self.assertEqual(list_robust_mean[1].shape, (2,))
        self.assertEqual(list_inv_cov_mat[1].shape, (2,2))
