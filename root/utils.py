import sklearn
import numpy
from root.model.variational_parameters import VariationalParameters


def generate_labels_pred(variational_parameters : VariationalParameters, Y):
    from jax import numpy as jnp
    labels_pred = []
    for i in range(Y.shape[0]):
        labels_pred.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))

    return labels_pred

def calculate_ARI(labels_true, labels_pred):
    return sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)


def save_results(hyperparameters_model, main_time, ARI, labels_pred):
    p = hyperparameters_model.p
    M = hyperparameters_model.M
    output_name = 'output_' + str(p) + '_' + str(M) + '.txt'

    line1 = "n\tp\tmain_time_secs\tari"

    line2 = str(hyperparameters_model.M) + '\t' + str(hyperparameters_model.p) + '\t' + \
            str(main_time) + '\t' + str(ARI)

    lines = [line1, line2]
    with open(output_name, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    labels_pred_name = 'labels_pred_' + str(p) + '_' + str(M) + '.csv'
    numpy.savetxt(labels_pred_name, labels_pred, delimiter=",")