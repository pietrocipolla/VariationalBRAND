from jax import numpy as jnp

class NIW:
    def __init__(self, mu, nu, lambdA, phi):
        self.mu = mu   #n_elem x p
        self.nu = nu   #n_elem
        self.lambdA = lambdA  #n_elem
        self.phi = phi  #n_elem x p x p
