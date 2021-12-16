from root.bin.save_load_numpy import load_data_nupy
from root.controller.cavi.elbo.elbo_calculator import elbo_calculator
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.cavi.updater.parameters_updater import update_parameters
from jax import numpy as jnp
from root.model.hyperparameters_model import HyperparametersModel
from root.model.user_input_model import UserInputModel
import matplotlib.pyplot as plt


def cavi(Y: jnp.array, hyperparameters_model : HyperparametersModel, user_input_parameters: UserInputModel):
    n_iter = user_input_parameters.n_iter
    tol = user_input_parameters.tol
    p = hyperparameters_model.p

    variational_parameters = init_cavi(user_input_parameters)
    elbo_values = []

    for i in range(n_iter):
        update_parameters(Y, hyperparameters_model, variational_parameters)
        elbo_values.append(elbo_calculator(Y, hyperparameters_model, variational_parameters, p))

        print('\nelbo: ',i, elbo_values,)
        #print(variational_parameters.toString())

        print('out: ', variational_parameters.sum_phi_k)

        # if (i > 0) & ((elbo_values[i]) ** 2 < tol):
        #     print('convergence of elbo')
        #     return variational_parameters, elbo_values

    X = load_data_nupy()
    ll = []
    for i in range(1000):
        ll.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))
    plt.scatter(X[:, 0], X[:, 1], c=ll, s=40, cmap='viridis')
    #plt.show()
    plt.savefig('figure.png')
    print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")

