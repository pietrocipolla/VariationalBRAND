from controller.sample_data_handler.data_generator import generate_sample_data, get_training_set
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters


def generate_sample_data_and_robust_parameters():
    num_clusters = 5
    num_clusters_training = 3
    num_samples = 500

    Y, labels = generate_sample_data(num_clusters, num_samples)
    Y_training, labels_training = get_training_set(Y, labels, num_clusters_training)

    robust_mean, robust_inv_cov_mat = calculate_robust_parameters(Y_training, labels_training, num_clusters_training)

    return Y, robust_mean, robust_inv_cov_mat