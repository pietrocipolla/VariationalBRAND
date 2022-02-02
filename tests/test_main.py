from main import var_brand
from unittest import TestCase
import os
import random
import numpy as np

# class Test(TestCase):
#     def test_main(self):
#         # SPECIFY DATASETS' INFO
#         Y_ALL_FILENAME = 'Y.csv'
#         Y_TRAINING_FILENAME = 'Y_training.csv'
#         LABELS_TRAINING_FILENAME = 'labels_training.csv'
#         LABELS_TOT_FILENAME = 'labels_tot.csv'
#
#         var_brand(Y_ALL_FILENAME, Y_TRAINING_FILENAME, LABELS_TRAINING_FILENAME)

class Test_Multiple(TestCase):
    def test_main(self):
        import sys
        FOLDER = sys.argv[0]  #"/home/eb/brand_tests/mcmc/
        SEED = sys.argv[1]
        random.seed(SEED)
        np.random.seed(SEED)

        file_list = os.listdir(FOLDER)

        #print(file_list)
        datasets = []

        labels_training_list = []

        for file in file_list:
            if 'labels_training' in file:
                labels_training_list.append(file)


        labels_training_list = sorted(labels_training_list ,key=lambda x: (float(x.split('_')[2]),int((x.split('_')[3]).split('.csv')[0])))
        #print(labels_training_list)



        for file in labels_training_list:
            temp_list = []
            temp = file.split('labels_training')[1]
            #print(temp)
            temp_training = 'Y_training' + temp
            if temp_training in file_list:
                Y_training = temp_training
                #print(Y_training)
                temp_Y = 'Y' + temp
                if temp_Y in file_list:
                    Y = temp_Y
                    #print(Y)
                    temp_labels_tot = 'labels_tot' + temp
                    if temp_labels_tot in file_list:
                        labels_tot = temp_labels_tot
                        # print(Y)
                        temp_list.append(FOLDER+Y)
                        temp_list.append(FOLDER+Y_training)
                        temp_list.append(FOLDER+file)
                        temp_list.append(FOLDER + labels_tot)
                        datasets.append(temp_list)

        print(datasets)

        for list in datasets:
            print(list[0],list[1], list[2])
            try:
                var_brand(Y_TOT_FILENAME=list[0], Y_TRAINING_FILENAME=list[1],
                          LABELS_TRAINING_FILENAME=list[2], LABELS_TOT_FILENAME=list[3])
            except Exception as e:
                print(e)



