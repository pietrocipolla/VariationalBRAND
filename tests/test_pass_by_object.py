from jax import numpy as jnp

def test_pass_by_object(array ):
    array.__setitem__(0,0)
    print('in :', array)

array = [1]
print('out pre:', array)
test_pass_by_object(array)
print('out post: ', array)

def test_pass_by_object(array : jnp.array ):
    array = array.at[0].set(0)
    print('in jax:', array)

array = jnp.array([1])
print('out pre jax:', array)
test_pass_by_object(array)
print('out post jax: ', array)

