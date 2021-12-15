from jax import numpy as jnp

from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.specify_user_input import specify_user_input
from model.variational_parameters import VariationalParameters

Y = generate_some_data_example()
Y_training, num_classes_training = get_training_set_example(Y)

list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)


def test_pass_by_object(array : VariationalParameters):
    array.a_k_beta = array.a_k_beta.at[0].set(0)
    print('in :', array.a_k_beta[0])

array : VariationalParameters = init_cavi(user_input_parameters)
print('VariationalParameters')
print('out pre:', array.a_k_beta[0])
test_pass_by_object(array)
print('out post: ', array.a_k_beta[0])


def test_pass_by_object(sum_phi_k):
    sum_phi_k = sum_phi_k.at[0].set(0)
    print('in :', sum_phi_k[0])

variational_parameters : VariationalParameters = init_cavi(user_input_parameters)
sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=0)
print('\nsum_phi_k')
print('out pre:', sum_phi_k[0])
test_pass_by_object(sum_phi_k)
print('out post: ', sum_phi_k[0])
