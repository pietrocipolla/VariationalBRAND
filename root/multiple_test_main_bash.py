import random
import numpy as np
import os
from main import var_brand

def multiple_test_main_bash():
    import sys
    FOLDER = sys.argv[1]  # "/home/eb/brand_tests/mcmc/
    SEED = int(sys.argv[2])

    random.seed(SEED)
    np.random.seed(SEED)

    file_list = os.listdir(FOLDER)

    # print(file_list)
    datasets = []

    labels_training_list = []

    for file in file_list:
        if 'labels_training' in file:
            labels_training_list.append(file)

    labels_training_list = sorted(labels_training_list,
                                  key=lambda x: (float(x.split('_')[3]), int((x.split('_')[4]).split('.csv')[0])))
    # print(labels_training_list)

    for file in labels_training_list:
        temp_list = []
        temp = file.split('labels_training')[1]
        # print(temp)
        temp_training = 'Y_training' + temp
        if temp_training in file_list:
            Y_training = temp_training
            # print(Y_training)
            temp_Y = 'Y' + temp
            if temp_Y in file_list:
                Y = temp_Y
                # print(Y)
                temp_labels_tot = 'labels_tot' + temp
                if temp_labels_tot in file_list:
                    labels_tot = temp_labels_tot
                    # print(Y)
                    temp_list.append(FOLDER + Y)
                    temp_list.append(FOLDER + Y_training)
                    temp_list.append(FOLDER + file)
                    temp_list.append(FOLDER + labels_tot)
                    temp_list.append(labels_tot.split('_')[2])
                    datasets.append(temp_list)

    # print("datasets")
    # print(datasets)

    for list in datasets:
        # print(list[0], list[1], list[2], list[3])
        try:
            var_brand(Y_TOT_FILENAME=list[0], Y_TRAINING_FILENAME=list[1],
                      LABELS_TRAINING_FILENAME=list[2], LABELS_TOT_FILENAME=list[3], SEED=list[4])
        except Exception as e:
            print(e)

multiple_test_main_bash()