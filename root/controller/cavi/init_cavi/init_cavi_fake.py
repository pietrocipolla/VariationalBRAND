from jax import numpy as jnp

from model.NIW import NIW
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters


def init_cavi_fake(hyperparameters_model: HyperparametersModel):
    M = hyperparameters_model.M
    J = hyperparameters_model.J
    T = hyperparameters_model.T

    phi_m_k_temp = jnp.zeros(M, J + T)

    for m in range(hyperparameters_model.M):
        for k in range(hyperparameters_model.J):
            phi_m_k_temp[m, k] = 1 / (J + 1)
        for k in range(J, J + T):
            phi_m_k_temp[m, k] = (1 / (J + 1)) * (0.5 ** (k - J)) * (1 / (1 - 0.5 ** T))

    return VariationalParameters(
        phi_m_k = phi_m_k_temp,

        eta_k = hyperparameters_model.a_dir_k,

        a_k_beta = jnp.ones(T-1),

        b_k_beta = jnp.multiply(jnp.ones(T-1),hyperparameters_model.gamma),

        nIW_MIX_VAR = hyperparameters_model.nIW_MIX_0,

        nIW_DP_VAR = NIW(mu=jnp.repeat(hyperparameters_model.nIW_DP_0.mu_0_DP, repeats = T, axis=0),
                                                nu=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.nu_0_DP),
                                                lambdA=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.lambda_0_DP),
                                                phi= jnp.repeat(hyperparameters_model.nIW_DP_0.phi_0_dp, repeats = T, axis=0))
        #todo check se output in ndarray da probemi (cfr funzione jnp.repeat)
    )


