
import numpy as np
from pandas import read_csv
from numpy import dstack
import h5py
import csv 
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing


#UCI

# load dataset

# load a single file as a numpy array

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_y(prefix + group + '/y_'+group+'.txt')
    p = load_y(prefix + group + '/subject_'+group+'.txt')
    return X, y, p

def one_hot_func(y):
    shape = (y.size, y.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(y.size)
    one_hot[rows, y] = 1
    return one_hot


# load all train
trainX, trainy, trainp = load_dataset('train', 'UCI_HAR_Dataset/')
trainy = trainy.reshape(-1)
trainp = trainp.reshape(-1)

#once all of the data has been read in and the files have been appended, save

# load all test
testX, testy, testp = load_dataset('test', 'UCI_HAR_Dataset/')
testy = testy.reshape(-1)
testp = testp.reshape(-1)

X = np.concatenate([trainX,testX],0)
Y = np.concatenate([trainy, testy])
P = np.concatenate([trainp, testp])
y_onehot= one_hot_func(Y)
p_onehot= one_hot_func(P)

print(X.shape, Y.shape, P.shape, y_onehot.shape, p_onehot.shape)
h5f = h5py.File('RL_acc_uci_data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('y', data=Y)
h5f.create_dataset('p', data=P)
h5f.create_dataset('y_onehot', data=y_onehot)
h5f.create_dataset('p_onehot', data=p_onehot)
h5f.close()
print(X.shape, Y.shape)
print(y_onehot.shape)
print(p_onehot.shape)
