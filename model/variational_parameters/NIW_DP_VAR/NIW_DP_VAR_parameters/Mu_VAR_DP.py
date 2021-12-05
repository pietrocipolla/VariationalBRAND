from jax import numpy as jnp

class Mu_VAR_DP:
    #-> matrice (Txp) -> riga per riga ci sono le medie delle componenti della misturaDP
    # (mu_var_DP[i,:] -> media della (i+1)-esima NIW della misturaDP)
    def __init__(self, mu_VAR_DP: jnp.ndarray):
        self.mu_VAR_DP = jnp.array(mu_VAR_DP)

# # example
# T = 2
# p = 2
# user_input = jnp.array([[2,3],[4,5]])
#
# hello = Mu_VAR_DP(user_input)
# print(hello.mu_VAR_DP)

