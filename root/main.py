from controller.sample_data_handler.sample_data_handler import generate_sample_data_and_robust_parameters
from controller.cavi.cavi import cavi
from model.variational_parameters.variational_parameters import VariationalParameters
from root.model.hyperparameters_model.NIW import NIW
from root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from jax import numpy as jnp

if __name__ == '__main__':
    ######### USER INPUT ##########
    ######  Y, R_MEAN, R_INV_COV
    Y, robust_mean, robust_inv_cov_mat = generate_sample_data_and_robust_parameters()

    mu_0_MIX = robust_mean
    phi_0_MIX = robust_inv_cov_mat

    ######### USER INPUT ##########
    ######  hyperparameters_model
    hyperparameters_model = HyperparametersModel(
        gamma=5,
        # gamma -> parametro dello Stick Breaking -> scalare
        # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior

        a_dir_k=jnp.ones(3 + 1),
        # a_dir_k -> vettore delle componenti della Dirichlet -> vettore di (J+1) componenti
        # J = num_classes_learning

        nIW_DP_0=NIW(
            mu=jnp.ones(2)[None, :],  # così che sia comunque della forma n_elems x p
            nu=jnp.array([2]),
            lambdA=jnp.array([1]),
            phi=jnp.identity(2)[None, :]
            # vettore (p) componenti, con p = 2
        ),

        nIW_MIX_0=NIW(
            mu=mu_0_MIX,
            nu=jnp.multiply(jnp.ones(3), 2),
            lambdA=jnp.ones(3),
            phi=phi_0_MIX,  # varianza #todo check coerenza robust estimaros
        )  # TODO trovare inizializzazione più furba
    )

    ######### USER INPUT ##########
    ######  variational_hyperparameters
    M = hyperparameters_model.M
    J = hyperparameters_model.J
    T = hyperparameters_model.T

    phi_m_k_temp = jnp.zeros(M, J + T)

    for m in range(hyperparameters_model.M):
        for k in range(hyperparameters_model.J):
            phi_m_k_temp[m, k] = 1 / (J + 1)
        for k in range(J, J + T):
            phi_m_k_temp[m, k] = (1 / (J + 1)) * (0.5 ** (k - J)) * (1 / (1 - 0.5 ** T))

    variational_parameters =  VariationalParameters(
        phi_m_k=phi_m_k_temp,

        eta_k=hyperparameters_model.a_dir_k,

        a_k_beta=jnp.ones(T - 1),

        b_k_beta=jnp.multiply(jnp.ones(T - 1), hyperparameters_model.gamma),

        nIW_MIX_VAR=hyperparameters_model.nIW_MIX_0,

        nIW_DP_VAR=NIW(mu=jnp.repeat(hyperparameters_model.nIW_DP_0.mu_0_DP, repeats=T, axis=0),
                       nu=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.nu_0_DP),
                       lambdA=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.lambda_0_DP),
                       phi=jnp.repeat(hyperparameters_model.nIW_DP_0.phi_0_dp, repeats=T, axis=0)),
    )
    # todo check se output in ndarray da probemi (cfr funzione jnp.repeat)

    # CAVI (update + elbo)
    n_iter = 1000
    #data all data
    variational_parameters, elbo_values = cavi(Y, hyperparameters_model,variational_parameters, n_iter)
