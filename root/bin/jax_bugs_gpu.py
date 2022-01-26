from jax._src.numpy.linalg import inv as jinv
from jax import numpy as jnp

def buggy_example():
    PHI_MIX = jnp.array([[[ 3.28045,0.17521389],
                        [ 0.17521389 , 5.1232452 ]],

                        [[ 4.6521783 , -0.8761855 ],
                        [-0.8761855  , 2.734903  ]],

                        [[ 2.9329915 , -0.18445067],
                        [-0.18445067 , 3.064779  ]]])

    out = jinv(PHI_MIX[0, :, :])


buggy_example()