from unittest import TestCase

from controller.cavi.utils.useful_functions import E_log_beta, E_log_dens_norm_inv_wish, E_log_dens_dir, E_log_norm
from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.user_input import specify_user_input
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters
from jax import numpy as jnp


class Test(TestCase):
    def test_e_log_beta(self):
        a = 2
        b = 3
        result = E_log_beta(a, b)
        # print(E_log_beta(a,b))
        self.assertEqual(result, -1.0833335)


class Test(TestCase):
    def test_e_log_dens_norm_inv_wish(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data

        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        mu = hyperparameters_model.nIW_DP_0.mu
        nu = hyperparameters_model.nIW_DP_0.nu
        lam = hyperparameters_model.nIW_DP_0.lambdA
        psi = hyperparameters_model.nIW_DP_0.phi
        p = hyperparameters_model.p

        out = E_log_dens_norm_inv_wish(mu, nu, lam, psi, p)
        # print (out)
        self.assertEqual(out, jnp.array([[0.895777]]))


class Test(TestCase):
    def test_e_log_dens_dir(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data

        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)
        eta = variational_parameters.eta_k
        J = hyperparameters_model.J

        out = E_log_dens_dir(eta, J)
        print(out)  # [0. 0. 0. 0.]
        print(type(out))  # <class 'jaxlib.xla_extension.DeviceArray'>
        self.assertEqual(out[0], 0.0)
        self.assertEqual(len(out), 4)


class Test(TestCase):
    def test_e_log_norm(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data

        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        mu = hyperparameters_model.nIW_DP_0.mu[0]
        nu = hyperparameters_model.nIW_DP_0.nu[0]
        lam = hyperparameters_model.nIW_DP_0.lambdA[0]
        psi = hyperparameters_model.nIW_DP_0.phi[0]
        p = hyperparameters_model.p

        out = E_log_norm(data[0],mu,nu,lam,psi,p)
        print(out)
