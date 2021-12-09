from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.cavi.updater.parameters_updater import update_parameters
from controller.cavi.elbo.elbo_calculator import elbo_calculator
from model.user_input_model import UserInputModel
from model.hyperparameters_model import HyperparametersModel
from jax import numpy as jnp


#Nota: aggiungere input dati qua dentro
#To do: fissare p

def cavi(data: jnp.array, hyperparameters_model : HyperparametersModel, user_input_parameters: UserInputModel, n_iter):
    variational_parameters = init_cavi(user_input_parameters)
    elbo_values = []
    #p=?

    for i in range(n_iter):
        variational_parameters = update_parameters(data, hyperparameters_model, variational_parameters)
        elbo_values.append(elbo_calculator(hyperparameters_model, variational_parameters, p))
        if (i > 0) & ((elbo_values[-1] - elbo_values[-2]) ** 2 < tol):
            print('convergence of elbo')
            return variational_parameters, elbo_values

    return variational_parameters, elbo_values