from model.variational_parameters import VariationalParameters


def generate_induced_partition(Y, robust_mean, variational_parameters : VariationalParameters):
    import matplotlib.pyplot as plt
    from jax import numpy as jnp
    ll = []
    for i in range(750):
        ll.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))
    plt.scatter(Y[:, 0], Y[:, 1], c=ll, s=40, cmap='viridis')
    plt.scatter(robust_mean[:, 0], robust_mean[:, 1], c='red', s=40, cmap='viridis')
    # plt.show()
    plt.savefig('figure.png')
    print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")