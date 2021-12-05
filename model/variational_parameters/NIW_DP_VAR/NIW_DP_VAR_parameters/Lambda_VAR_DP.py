from jax import numpy as jnp

class Lambda_VAR_DP:
    #-> vettore (T) componenti
    # (lambda_var_DP[i] = lambda_var della (i+1) esima NIW della misturaDP)
    def __init__(self, lambda_VAR_DP: jnp.array):
        self.lambda_VAR_DP = lambda_VAR_DP

# example
# T = 2
# user_input = jnp.ones(T)
# hello = Lambda_VAR_DP(user_input)
# print(hello.lambda_VAR_DP)
