# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

def save_data_numpy():
    # define data
    data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # save to csv file
    savetxt('data.csv', data, delimiter=',')

def load_data_nupy():
    # load numpy array from csv file
    from numpy import loadtxt
    # load array
    data = loadtxt('data.csv', delimiter=',')
    # print the array
    print(data)