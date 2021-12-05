from jax import numpy as jnp

class Nu_0_MIX:
    #-> vettore (J) componenti
    #   (nu_0_MIX[i] = nu_0 della (i+1) esima NIW della mistura)

    def __init__(self, nu_0_MIX : jnp.array):
        self.nu_0_MIX = jnp.array(nu_0_MIX)

#example
# J = 3
# user_input = jnp.ones(J)
# hello = nu_0_MIX(user_input)
# print(hello.nu_0_MIX)
