from jax import numpy as jnp

class Lambda_0_MIX:
    #-> vettore (J) componenti
    #   (lambda_0_MIX[i] = lambda_0 della (i+1)  esima NIW della mistura)

    def __init__(self, lambda_0_MIX: jnp.array):
        self.nu_0_MIX = lambda_0_MIX

# example
# J = 3
# user_input = jnp.ones(J)
# hello = Lambda_0_MIX(user_input)
# print(hello.nu_0_MIX)
