from jax import numpy as jnp

class Eta_k:
    #-> parametri della dirichlet -> vettore (J+1) componenti
    def __init__(self, eta_k : jnp.array):
        self.eta_k = jnp.array(eta_k)

# #example
# J = 3
# user_input = jnp.ones(J+1)
# hello = Eta_k(user_input)
# print(hello.eta_k)

