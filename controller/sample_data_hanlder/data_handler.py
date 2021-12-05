from controller.sample_data_hanlder.data_generator import get_learning_set, get_test_set, \
    generate_sample_data

def generate_learning_and_test_sets():
    num_classes = 5
    n_samples = 500
    num_classes_learning = 3

    X, labels, num_classes = generate_sample_data(num_classes, n_samples)

    return get_learning_set(X, labels, num_classes, num_classes_learning), \
           get_test_set(X, labels, num_classes, num_classes_learning), \
           num_classes, n_samples


#print(generate_learning_and_test_sets())