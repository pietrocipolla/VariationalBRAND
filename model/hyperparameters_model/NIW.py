from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters import Lambda_0_DP, Nu_0_DP, PHI_0_DP
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.Mu_0_DP import Mu_0_DP
from jax import numpy as jnp

class NIW:

    def __init__(self, mu_0: jnp.array, nu_0 : jnp.array, lambda_0 : jnp.array, phi_0 : jnp.array):
        self.mu_0 = mu_0   #n_elem x p
        self.nu_0 = nu_0   #n_elem
        self.lambda_0 = lambda_0  #n_elem
        self.phi_0 = phi_0  #n_elem x p x p
