from jax import numpy as jnp
from model.NIW import NIW

class VariationalParameters:
    def __init__(self, phi_m_k: jnp.array, eta_k: jnp.array, a_k_beta: jnp.array, b_k_beta: jnp.array,
                 nIW_VAR : NIW):
        self.phi_m_k = phi_m_k
        self.eta_k = eta_k
        self.a_k_beta = a_k_beta
        self.b_k_beta = b_k_beta
        self.nIW : NIW = nIW_VAR
