from model.variational_parameters.variational_model import VariationalParameters


def update_parameters(variational_parameters : VariationalParameters):
    #update paramerters
    #example
    variational_parameters.eta_k.eta_k = variational_parameters.eta_k.eta_k + 3
    #update jax array
    #https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/jax_for_the_impatient.ipynb
    #x.at[0,:].set(3.0)
    return variational_parameters