from jax import numpy as jnp

class B_k_beta:
    # V_i ~ Beta(a_k[i-1], b_k[i-1])
    #Secondo parametro delle beta -> vettore (T-1) componenti
    # #V_T, ultima beta variazionale, è SEMPRE 1
    def __init__(self, b_k : jnp.array):
        self.b_k = b_k

# #example
# T = 2
# user_input = jnp.ones(T-1)
# hello = B_k(user_input)
# print(hello.b_k)

