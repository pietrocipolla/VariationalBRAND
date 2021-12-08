from jax import numpy as jnp

class A_dir_k:
#a_dir_k -> vettore delle componenti della Dirichlet
#           -> vettore di (J+1) componenti
#cambiare notazione, utilizziamo ak anche come primo parametro delle Beta variazionali
    def __init__(self, a_dir_k : jnp.array):
        self.a_dir_k = a_dir_k

#example
# J = 3
# user_input = [1, 1, 1, 1]
# hello = A_dir_k(user_input)
# print(hello.a_dir_k)
#
# user_input = jnp.ones(J+1)
# hello = A_dir_k(user_input)
# print(hello.a_dir_k)