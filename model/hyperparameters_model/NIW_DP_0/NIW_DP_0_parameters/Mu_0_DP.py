from jax import numpy as jnp

class Mu_0_DP:
    #vettore (p) componenti
    def __init__(self, mu_0_DP : jnp.array):
        self.mu_0_DP = jnp.array(mu_0_DP)

#example
# p = 2
# user_input = jnp.ones(p)
# hello = Mu_0_DP(user_input)
# print(hello.mu_0_DP)

