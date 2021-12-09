from jax import numpy as jnp
from model import NIW

class VariationalParameters:
    def __init__(self, phi_m_k: jnp.array, eta_k: jnp.array, a_k_beta: jnp.array, b_k_beta: jnp.array,
                 nIW_DP_VAR : NIW, nIW_MIX_VAR : NIW):
        self.phi_m_k = phi_m_k
        self.eta_k = eta_k
        self.a_k_beta = a_k_beta
        self.b_k_beta = b_k_beta
        self.nIW_MIX_VAR = nIW_MIX_VAR
        self.nIW_DP_VAR = nIW_DP_VAR
