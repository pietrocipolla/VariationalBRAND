from jax import numpy as jnp

class Mu_0_MIX:
    #mu_0_MIX -> matrice (Jxp) -> riga per riga ci sono le medie delle componenti della mistura
    #   (mu_0_MIX[i,:] -> media della (i+1)-esima NIW della mistura)
    def __init__(self, Mu_0_MIX: jnp.ndarray):
        self.Mu_0_MIX = Mu_0_MIX

# # example
# J = 3
# p = 2
# user_input = jnp.array([[2,3],[4,5],[1,2]])
#
# hello = Mu_0_MIX(user_input)
# print(hello.Mu_0_MIX)


