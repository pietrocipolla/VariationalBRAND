from jax import numpy as jnp

class NIW:
    def __init__(self, mu: jnp.array, nu : jnp.array, lambdA : jnp.array, phi : jnp.array):
        self.mu = mu   #n_elem x p
        self.nu = nu   #n_elem
        self.lambdA = lambdA  #n_elem
        self.phi = phi  #n_elem x p x p
