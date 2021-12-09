from controller.cavi.elbo.elbo_calculator import elbo_calculator
from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.cavi.updater.parameters_updater import update_parameters
from jax import numpy as jnp
from model.hyperparameters_model import HyperparametersModel
from model.user_input_model import UserInputModel

def cavi(Y: jnp.array, hyperparameters_model : HyperparametersModel, user_input_parameters: UserInputModel):
    n_iter = user_input_parameters.n_iter
    tol = user_input_parameters.tol
    p = Y.shape[1] #number of data coordinates

    variational_parameters = init_cavi(user_input_parameters)
    elbo_values = []

    for i in range(n_iter):
        variational_parameters = update_parameters(Y, hyperparameters_model, variational_parameters)
        elbo_values.append(elbo_calculator(Y, hyperparameters_model, variational_parameters, p))
        if (i > 0) & ((elbo_values[-1] - elbo_values[-2]) ** 2 < tol):
            print('convergence of elbo')
            return variational_parameters, elbo_values

    return variational_parameters, elbo_values