from jax import numpy as jnp

class Mu_VAR_MIX:
    #-> matrice (Jxp) -> riga per riga ci sono le medie delle componenti della mistura
    # (mu_var_MIX[i,:] -> media della (i+1)-esima NIW della mistura)
    def __init__(self, Mu_VAR_MIX: jnp.ndarray):
        self.Mu_VAR_MIX = Mu_VAR_MIX

# # example
# J = 3
# p = 2
# user_input = jnp.array([[2,3],[4,5],[1,2]])
#
# hello = Mu_VAR_MIX(user_input)
# print(hello.Mu_VAR_MIX)


