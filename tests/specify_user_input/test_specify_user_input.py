from unittest import TestCase

import numpy

from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input


class Test(TestCase):
    def test_specify_user_input(self):
        #HYPERPARAMETERS
        Y = generate_some_data_example()
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

        user_input = specify_user_input(list_robust_mean, list_inv_cov_mat)

        self.assertEqual(user_input.J, 3)
        self.assertEqual(user_input.T, 5)
        self.assertEqual(user_input.M, 500)

        self.assertEqual(user_input.gamma, 5)
        self.assertEqual(len(user_input.eta_k), 4)

        #O_DP
        self.assertEqual(len(user_input.mu_0_DP), 2)
        self.assertEqual(user_input.nu_0_DP.shape, (1,))
        self.assertEqual(user_input.lambda_0_DP.shape, (1, ))
        self.assertEqual(user_input.PHI_0_DP.shape, (2,2))

        #0_MIX
        self.assertEqual(len(user_input.mu_0_MIX), 3)
        self.assertEqual(len(user_input.mu_0_MIX[0]), 2)
        self.assertEqual(len(user_input.nu_0_MIX), 3)
        self.assertEqual(len(user_input.lambda_0_MIX), 3)
        self.assertEqual(len(user_input.PHI_0_MIX), 3)
        self.assertEqual(user_input.PHI_0_MIX[0].shape, (2, 2))

        #VARIATIONAL HYPERPARAMETERS
        self.assertEqual(user_input.Phi_m_k.shape, (500, 8)) #todo check coerenza con example 3+2
        self.assertEqual(len(user_input.eta_k), 4)

        self.assertEqual(len(user_input.a_k_beta), 4)
        self.assertEqual(len(user_input.b_k_beta), 4)

        #NIW_DP_VAR
        self.assertEqual(len(user_input.mu_var_DP), 5)
        self.assertEqual(user_input.mu_var_DP[0].shape, (2,))
        # print('mu_var_DP', user_input.mu_var_DP)

        self.assertEqual(len(user_input.nu_var_DP), 5)

        self.assertEqual(len(user_input.lambda_var_DP), 5)

        self.assertEqual(len(user_input.PHI_var_DP), 5)
        self.assertEqual(user_input.PHI_var_DP[0].shape, (2,2))

        #NIW_MIX_VAR
        self.assertEqual(len(user_input.mu_VAR_MIX), 3)
        self.assertEqual(user_input.mu_VAR_MIX[0].shape, (2,))

        self.assertEqual(user_input.nu_VAR_MIX.shape, (3,))

        self.assertEqual(len(user_input.lambda_VAR_MIX), 3)

        self.assertEqual(len(user_input.PHI_VAR_MIX), 3)
        self.assertEqual(user_input.PHI_VAR_MIX[0].shape, (2, 2))







