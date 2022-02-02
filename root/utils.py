import sklearn
import numpy
from model.variational_parameters import VariationalParameters


def generate_labels_pred(variational_parameters : VariationalParameters, Y):
    from jax import numpy as jnp
    labels_pred = []
    for i in range(Y.shape[0]):
        labels_pred.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))

    return labels_pred

def calculate_ARI(labels_true, labels_pred):
    return sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)


def save_results(hyperparameters_model, main_time_secs, n_iter, ARI, labels_pred,SEED):
    p = hyperparameters_model.p
    M = hyperparameters_model.M

    main_time_mins = float(main_time_secs) / 60
    output_name = 'output_' + str(SEED) + '_' + str(p) + '_' + str(M) + '.txt'

    line1 = "n\tp\tn_iter\tmain_time_secs\tmain_time_mins\tARI"

    line2 = str(hyperparameters_model.M) + '\t' + str(hyperparameters_model.p) + '\t' + \
            str(n_iter) + '\t' + str(main_time_secs) + '\t' + str(main_time_mins) + '\t' + str(ARI)

    lines = [line1, line2]
    with open(output_name, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    output_csv_name = 'output_csv_' + str(SEED) + '_' + str(p) + '_' + str(M) + '_.txt'

    to_save = [str(SEED),str(hyperparameters_model.M), str(hyperparameters_model.p),
            str(n_iter) ,str(main_time_secs),str(main_time_mins) , str(ARI)]

    numpy.savetxt(output_csv_name, to_save, delimiter=",", fmt="%s")

    labels_pred_name = 'labels_pred_' + str(SEED) + '_'+ str(p) + '_' + str(M) + '.csv'
    numpy.savetxt(labels_pred_name, labels_pred, delimiter=",")

