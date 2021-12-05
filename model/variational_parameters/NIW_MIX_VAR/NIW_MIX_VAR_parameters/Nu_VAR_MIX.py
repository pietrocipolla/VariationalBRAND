from jax import numpy as jnp

class Nu_VAR_MIX:
    #-> vettore (J) componenti
    #   (nu_VAR_MIX[i] = nu_VAR della (i+1) esima NIW della mistura)

    def __init__(self, nu_VAR_MIX : jnp.array):
        self.nu_VAR_MIX = jnp.array(nu_VAR_MIX)

#example
# J = 3
# user_input = jnp.ones(J)
# hello = Nu_VAR_MIX(user_input)
# print(hello.nu_VAR_MIX)
