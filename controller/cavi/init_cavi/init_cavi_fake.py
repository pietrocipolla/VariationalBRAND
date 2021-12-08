from jax import numpy as jnp

from model.hyperparameters_model.NIW import NIW
from model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR import NIW_DP_VAR
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Lambda_VAR_DP import Lambda_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Mu_VAR_DP import Mu_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Nu_VAR_DP import Nu_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.PHI_VAR_DP import PHI_VAR_DP
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR import NIW_MIX_VAR
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Lambda_VAR_MIX import Lambda_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Mu_VAR_MIX import Mu_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Nu_VAR_MIX import Nu_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.PHI_VAR_MIX import PHI_VAR_MIX
from model.variational_parameters.variational_parameters import VariationalParameters
from model.variational_parameters.a_k_beta import A_k_beta
from model.variational_parameters.b_k_beta import B_k_beta
from model.variational_parameters.eta_k import Eta_k
from model.variational_parameters.phi_m_k import Phi_m_k

def init_cavi_fake(hyperparameters_model: HyperparametersModel, variational_parameters: VariationalParameters):
    M = hyperparameters_model.M
    J = hyperparameters_model.J
    T = hyperparameters_model.T

    phi_m_k_temp = jnp.zeros(M,J+T)

    for m in range(hyperparameters_model.M):
        for k in range(hyperparameters_model.J):
            phi_m_k_temp[m,k] = 1 / (J + 1)
        for k in range(J,J+T):
            phi_m_k_temp[m,k] = (1 / (J + 1))*(0.5**(k-J))*(1/(1 - 0.5**T))

    variational_parameters.phi_m_k = phi_m_k_temp

    variational_parameters.eta_k = hyperparameters_model.a_dir_k

    variational_parameters.a_k_beta = jnp.ones(T-1)

    variational_parameters.b_k_beta = jnp.multiply(jnp.ones(T-1),hyperparameters_model.gamma)

    variational_parameters.nIW_MIX_VAR = hyperparameters_model.nIW_MIX_0

    variational_parameters.nIW_DP_VAR = NIW(mu_0=jnp.repeat(hyperparameters_model.nIW_DP_0.mu_0_DP, repeats = T, axis=0),
                                            nu_0=jnp.multiply(jnp.ones(T),hyperparameters_model.nIW_DP_0.nu_0_DP),
                                            lambda_0=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.lambda_0_DP),
                                            phi_0 = jnp.repeat(hyperparameters_model.nIW_DP_0.phi_0_dp, repeats = T, axis=0))
    #todo check se output in ndarray da probemi (cfr funzione jnp.repeat)



