from jax import numpy as jnp

class PHI_0_MIX:
    # -> vettore di matrici (Jxpxp), sostanzialmente un ndarray ->
    #   PHI_0_MIX[i,:,:] = Matrice PHI della (i+1)esima NIW della mistura

    def __init__(self, PHI_0_MIX: jnp.ndarray):
        self.PHI_0_MIX = PHI_0_MIX

# # example
# J = 3
# p = 2
# user_input = jnp.array([
#                         [[2,3],[4,5],[1,2]],
#                         [[2,3],[4,5],[1,2]]])
#
# hello = PHI_0_MIX(user_input)
# print(hello.PHI_0_MIX)
