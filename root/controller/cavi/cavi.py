from root.controller.cavi.elbo.elbo_calculator import elbo_calculator
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.cavi.updater.parameters_updater import update_parameters
from jax import numpy as jnp
from root.model.hyperparameters_model import HyperparametersModel
from root.model.user_input_model import UserInputModel

def cavi(Y: jnp.array, hyperparameters_model : HyperparametersModel, user_input_parameters: UserInputModel):
    n_iter = user_input_parameters.n_iter
    tol = user_input_parameters.tol
    p = hyperparameters_model.p

    variational_parameters = init_cavi(user_input_parameters)
    starting_parameters = init_cavi(user_input_parameters)
    elbo_values = []

    i = 0
    stop = False

    print('\n')
    while ((i<n_iter) and (stop == False)):
        variational_parameters = update_parameters(Y, hyperparameters_model, variational_parameters, starting_parameters)
        elbo_values.append(elbo_calculator(Y, hyperparameters_model, variational_parameters, p))
        print('iter' ,i,' elbo: ',elbo_values[i])

        if (i > 0) and (abs(elbo_values[i] - elbo_values[i-1]) < tol):
            print('\nConvergence of elbo in ', i, ' iterations')
            stop = True

        i += 1

    print('\n')
    print('mu = ', variational_parameters.nIW.mu)
    print('lambda = ', variational_parameters.nIW.lambdA)
    print('nu = ', variational_parameters.nIW.nu)
    print('PHI = ', variational_parameters.nIW.phi)
    print('a_beta = ', variational_parameters.a_k_beta)
    print('b_beta = ', variational_parameters.b_k_beta)
    print('phi_m = ', variational_parameters.phi_m_k)
    print('eta = ', variational_parameters.eta_k)

    return variational_parameters