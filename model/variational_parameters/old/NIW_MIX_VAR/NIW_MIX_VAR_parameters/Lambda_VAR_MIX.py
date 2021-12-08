from jax import numpy as jnp

class Lambda_VAR_MIX:
    #-> vettore (J) componenti
    # (lambda_var_MIX[i] = lambda_var della (i+1) esima NIW della mistura)

    def __init__(self, lambda_VAR_MIX: jnp.array):
        self.lambda_VAR_MIX = lambda_VAR_MIX

# example
# J = 3
# user_input = jnp.ones(J)
# hello = Lambda_VAR_MIX(user_input)
# print(hello.lambda_VAR_MIX)
