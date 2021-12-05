from jax import numpy as jnp

class Nu_VAR_DP:
    #-> vettore (T) componenti (nu_var_MIX[i] = nu_var della (i+1) esima NIW della misturaDP)
    def __init__(self, nu_VAR_DP: jnp.array):
        self.nu_VAR_DP = jnp.array(nu_VAR_DP)

# example
# T = 2
# user_input = jnp.ones(T)
# hello = Nu_VAR_DP(user_input)
# print(hello.nu_VAR_DP)

