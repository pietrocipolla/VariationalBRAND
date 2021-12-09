from controller.sample_data_handler.sample_data_handler import generate_sample_data_and_robust_parameters
from controller.cavi.cavi import cavi
from controller.user_input import user_input
from model.hyperparameters_model.set_hyperparameters import set_hyperparameters
from model.variational_parameters.variational_parameters import VariationalParameters
from root.model.hyperparameters_model.NIW import NIW
from root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from jax import numpy as jnp

if __name__ == '__main__':
    #STEP 1
    #modify y, robust mean and robust inv_cov_mat to insert your data
    Y, robust_mean, robust_inv_cov_mat = generate_sample_data_and_robust_parameters()

    mu_0_MIX = robust_mean
    phi_0_MIX = robust_inv_cov_mat

    #modify user_input to change hyperparameters
    user_input_parameters = user_input(robust_mean, robust_inv_cov_mat)

    #STEP 2
    hyperparameters_model = set_hyperparameters(user_input_parameters)

    # CAVI (update + elbo)
    n_iter = 1000
    #data all data
    variational_parameters, elbo_values = cavi(Y, hyperparameters_model,user_input_parameters, n_iter)
