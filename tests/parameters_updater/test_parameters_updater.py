from unittest import TestCase

from numpy import loadtxt

from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.cavi.updater.parameters_updater import update_parameters, create_mask, update_dirichlet, \
    update_beta, update_NIW_mu, update_NIW_lambda, update_NIW_nu, update_NIW_PHI, update_phi_mk, init_update_NIW, \
    init_update_parameters, update_NIW
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters


class Test(TestCase):
    def test_parameters_updater(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        print(variational_parameters.toString())
        update_parameters(Y, hyperparameters_model, variational_parameters)
        print(variational_parameters.toString())

class Test(TestCase):
    def test_init_update_parameters(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        init_update_parameters(hyperparameters_model, variational_parameters)

        print(variational_parameters.sum_phi_k)
        print(variational_parameters.sum_phi_k.shape)
        print(variational_parameters.T_true)


class Test(TestCase):
    def test_update_dirichlet(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        init_update_parameters(hyperparameters, variational_parameters)

        print(variational_parameters.eta_k)
        print(variational_parameters.eta_k.shape)

        update_dirichlet(variational_parameters, hyperparameters)

        print(variational_parameters.eta_k)
        print(variational_parameters.eta_k.shape)


class Test(TestCase):
    def test_update_beta(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        init_update_parameters(hyperparameters, variational_parameters)

        T = hyperparameters.T
        print(variational_parameters.a_k_beta[T])
        print(variational_parameters.b_k_beta[T])

        update_beta(variational_parameters, hyperparameters)

        print(variational_parameters.a_k_beta)
        print(variational_parameters.a_k_beta.shape)


class Test(TestCase):
    def test_update_niw(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        update_NIW(data, variational_parameters, hyperparameters)


class Test(TestCase):
    def test_init_update_niw(self):
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        init_update_NIW(variational_parameters)

        print(variational_parameters.sum_y_phi)
        print(variational_parameters.sum_y_phi.shape)
        print(variational_parameters.y_bar)
        print(variational_parameters.y_bar.shape)


class Test(TestCase):
    def test_update_niw_mu(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        # print(variational_parameters.nIW_MIX_VAR.mu.shape)
        # print(variational_parameters.nIW_MIX_VAR.mu)
        #
        # print(variational_parameters.nIW_DP_VAR.mu.shape)
        # print(variational_parameters.nIW_DP_VAR.mu)

        init_update_NIW(variational_parameters)
        update_NIW_mu(variational_parameters, hyperparameters)

        # print(variational_parameters.nIW_MIX_VAR.mu.shape)
        # print(variational_parameters.nIW_MIX_VAR.mu)
        #
        # print(variational_parameters.nIW_DP_VAR.mu.shape)
        # print(variational_parameters.nIW_DP_VAR.mu)


class Test(TestCase):
    def test_update_niw_lambda(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        # print(variational_parameters.nIW_MIX_VAR.lambdA.shape)
        # print(variational_parameters.nIW_MIX_VAR.lambdA)
        #
        # print(variational_parameters.nIW_DP_VAR.lambdA.shape)
        # print(variational_parameters.nIW_DP_VAR.lambdA)

        init_update_NIW(variational_parameters)
        update_NIW_lambda(variational_parameters, hyperparameters)

        # print(variational_parameters.nIW_MIX_VAR.lambdA.shape)
        # print(variational_parameters.nIW_MIX_VAR.lambdA)
        #
        # print(variational_parameters.nIW_DP_VAR.lambdA.shape)
        # print(variational_parameters.nIW_DP_VAR.lambdA)


class Test(TestCase):
    def test_update_niw_nu(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        print(variational_parameters.nIW_MIX_VAR.nu.shape)
        print(variational_parameters.nIW_MIX_VAR.nu)

        print(variational_parameters.nIW_DP_VAR.nu.shape)
        print(variational_parameters.nIW_DP_VAR.nu)

        init_update_NIW(variational_parameters)
        update_NIW_nu(variational_parameters, hyperparameters)

        print(variational_parameters.nIW_MIX_VAR.nu.shape)
        print(variational_parameters.nIW_MIX_VAR.nu)

        print(variational_parameters.nIW_DP_VAR.nu.shape)
        print(variational_parameters.nIW_DP_VAR.nu)


class Test(TestCase):
    def test_update_niw_phi(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        print(variational_parameters.nIW_MIX_VAR.nu.shape)
        print(variational_parameters.nIW_MIX_VAR.nu)

        print(variational_parameters.nIW_DP_VAR.nu.shape)
        print(variational_parameters.nIW_DP_VAR.nu)

        print(variational_parameters.nIW_MIX_VAR.phi.shape)
        print(variational_parameters.nIW_MIX_VAR.phi)

        print(variational_parameters.nIW_DP_VAR.phi.shape)
        print(variational_parameters.nIW_DP_VAR.phi)

        update_NIW_PHI(data, variational_parameters, hyperparameters)

        print(variational_parameters.nIW_MIX_VAR.phi.shape)
        print(variational_parameters.nIW_MIX_VAR.phi)

        print(variational_parameters.nIW_DP_VAR.phi.shape)
        print(variational_parameters.nIW_DP_VAR.phi)


class Test(TestCase):
    def test_update_phi_mk(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        print(variational_parameters.phi_m_k.shape)
        update_phi_mk(data, variational_parameters, hyperparameters)
        print(variational_parameters.phi_m_k.shape)


# class Test(TestCase):
#     def test_create_tensor(self):
#         # todo


