from jax import numpy as jnp
from root.model.NIW import NIW

class VariationalParameters:
    def __init__(self, phi_m_k: jnp.array, eta_k: jnp.array, a_k_beta: jnp.array, b_k_beta: jnp.array,
                 nIW_DP_VAR : NIW, nIW_MIX_VAR : NIW):
        self.phi_m_k = phi_m_k
        self.eta_k = eta_k
        self.a_k_beta = a_k_beta
        self.b_k_beta = b_k_beta
        self.nIW_MIX_VAR : NIW = nIW_MIX_VAR
        self.nIW_DP_VAR : NIW = nIW_DP_VAR

    def toString(self):
        out = 'nIW_MIX_VAR.mu: \n'
        out += jnp.array_str(self.nIW_MIX_VAR.mu)
        out += '\nnIW_DP_VAR.mu: \n'
        out += jnp.array_str(self.nIW_DP_VAR.mu)
        out += '\nphi_m_k: \n'
        out += jnp.array_str(self.phi_m_k, max_line_width = 1000)
        jnp.set_printoptions(edgeitems = 10)
        return out