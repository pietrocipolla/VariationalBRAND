from unittest import TestCase
from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example


class Test(TestCase):
    def test_mcd(self):
        Y = generate_some_data_example()
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

        print(list_robust_mean)
        # print(list_inv_cov_mat)

        self.assertEqual(list_robust_mean[1].shape, (2,))
        self.assertEqual(list_inv_cov_mat[1].shape, (2, 2))

