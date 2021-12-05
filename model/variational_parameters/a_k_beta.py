from jax import numpy as jnp

class A_k_beta:
    #V_i ~ Beta(a_k[i-1], b_k[i-1])
    #primo parametro delle beta -> vettore (T-1) componenti
    # #V_T, ultima beta variazionale, è SEMPRE 1
    def __init__(self, a_k : jnp.array):
        self.a_k = a_k

# #example
# T = 2
# user_input = jnp.ones(T-1)
# hello = A_k(user_input)
# print(hello.a_k)

