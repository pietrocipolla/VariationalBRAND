from model.variational_parameters import VariationalParameters
from root.bin.save_load_numpy import load_data_nupy
from root.controller.cavi.elbo.elbo_calculator import elbo_calculator
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.cavi.updater.parameters_updater import update_parameters, init_update_parameters
from jax import numpy as jnp
from root.model.hyperparameters_model import HyperparametersModel
from root.model.user_input_model import UserInputModel


def cavi(Y: jnp.array, hyperparameters_model : HyperparametersModel, user_input_parameters: UserInputModel):
    n_iter = user_input_parameters.n_iter
    tol = user_input_parameters.tol
    p = hyperparameters_model.p

    variational_parameters : VariationalParameters = init_cavi(user_input_parameters)
    elbo_values = []

    for i in range(n_iter):
        init_update_parameters(hyperparameters_model, variational_parameters)
        update_parameters(Y, hyperparameters_model, variational_parameters)
        elbo_values.append(elbo_calculator(Y, hyperparameters_model, variational_parameters, p))

        print('\nelbo: ',i, elbo_values,)

        print('out: ', variational_parameters.sum_phi_k)

        # if (i > 0) & ((elbo_values[-1]-elbo_values[-2]) ** 2 < tol):
        #     print('convergence of elbo')
        #     return variational_parameters, elbo_values

    return variational_parameters

