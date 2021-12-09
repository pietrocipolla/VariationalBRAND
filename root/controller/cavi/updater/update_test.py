from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters


def update_parameters(data, hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters):
    variational_parameters.nIW_MIX_VAR.