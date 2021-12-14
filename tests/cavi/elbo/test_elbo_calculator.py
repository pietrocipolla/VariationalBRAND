from unittest import TestCase
from root.controller.cavi.elbo.elbo_calculator import elbo_calculator
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.cavi.updater.parameters_updater import update_parameters
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters


class Test(TestCase):
    def test_elbo_calculator(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        update_parameters(data, hyperparameters_model, variational_parameters)

        result = elbo_calculator(Y, hyperparameters_model, variational_parameters, Y.shape[1])
        #print(result)

        #print con update
        # MATRICIONAAA (500, 500)
        # f1 f2 done -0.04487786 -1900.0647
        # f3 f4 done nan nan #todo error solo se chiamato update parameters
        # f5 done -1.884512
        # f6 done -1354.237530708313
        # f7 done 0.0
        # f8 done -6.945502281188965

        #print senza update
        # WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
        # f1 f2 done -7284.827 -8381.909
        # f3 f4 done 36.40737 4.4788847
        # f5 done 0.0
        # f6 done -1008.870735168457
        # f7 done 0.0
        # f8 done -3.199993133544922

