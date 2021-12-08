from jax import numpy as jnp

class PHI_VAR_MIX:
    # -> vettore di matrici (Jxpxp), sostanzialmente un ndarray
    # -> PHI_var_MIX[i,:,:] = Matrice PHI della (i+1)esima NIW della mistura

    def __init__(self, PHI_VAR_MIX: jnp.ndarray):
        self.PHI_VAR_MIX = PHI_VAR_MIX

# # example
# J = 3
# p = 2
# user_input = jnp.array([
#                         [[2,3],[4,5],[1,2]],
#                         [[2,3],[4,5],[1,2]]])
#
# hello = PHI_VAR_MIX(user_input)
# print(hello.PHI_VAR_MIX)
