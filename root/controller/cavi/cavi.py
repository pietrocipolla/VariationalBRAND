from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.cavi.init_cavi.init_cavi_fake import init_cavi_fake
from controller.cavi.updater.parameters_updater import update_parameters
from root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters.variational_parameters import VariationalParameters
from jax import numpy as jnp

#Nota: aggiungere input dati qua dentro
def cavi(data: jnp.array, hyperparameters_model : HyperparametersModel,variational_parameters : VariationalParameters, n_iter):
    variational_parameters = init_cavi(hyperparameters_model)
    elbo_values = []

    for i in n_iter:
        variational_parameters = update_parameters(data, hyperparameters_model, variational_parameters)
        #elbo_values.append(elbo_calculator(dirichlet_hyperparameters_ai, eta_0 ,a_0,b_0, beta_0, fi_0, lambda_0, mu_0, nu_0, psi_0, gamma))
        #print elbo_values to check convergence?

    return variational_parameters, elbo_values