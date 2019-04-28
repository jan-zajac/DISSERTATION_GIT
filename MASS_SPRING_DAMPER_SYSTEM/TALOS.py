# Code for using Talos to carry out hyperparameter optimisation, for information on how to analyse output csv
# file, look at user manual at https://github.com/autonomio/talos
import sys
sys.path.insert(0, '../tools')
import tools
from keras import models
from keras import layers
import numpy as np
import talos as ta
from talos.model.normalizers import lr_normalizer
from keras.optimizers import rmsprop
from keras.regularizers import l1
##############################################################################

# read in data
data_filename = 'ODE_sols.txt'
label_filename = 'targets.txt'

# read in data
data = tools.read_data(data_filename)
labels = tools.read_data(label_filename)

##############################################################################
# simple splitting of training and testing data
# setting ratio dictating how we are going to divide the data into training and testing
ratio = 0.8
train_data, test_data, train_labels, test_labels = tools.data_split(data,labels, ratio)
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)
##############################################################################

# standardising the input data to have a mean = 0, and std = 1, and applying to the test data
# operation on training data

train_data_mean = train_data.mean(axis = 0)
train_data -= train_data_mean
test_data -= train_data_mean

train_data_std = train_data.std(axis = 0)
train_data /= train_data_std
test_data /= train_data_std

# changing all the values at t = 0 to 0s because they are nan
train_data[:,0] = 0
test_data[:,0] = 0
##############################################################################

# here we are standardising the targets in the same way
target_mean = train_labels.mean(axis = 0)
train_labels -= target_mean
test_labels -= target_mean

target_std = train_labels.std(axis = 0)
train_labels /= target_std
test_labels /= target_std

##############################################################################

# building of model
# anything with parameters in front of it is going to be scanned according to the parameters dictionary

def my_model(partial_train_data, partial_train_labels, val_train_data, val_train_labels, params):

    model = models.Sequential()
    model.add(layers.Dense(params['first_neuron'], activation = 'relu', input_shape=(train_data.shape[1],),
                           kernel_initializer='normal', kernel_regularizer = l1(params['weight_regularizer_1'])))
    model.add(layers.Dense(params['second_neuron'], activation = 'relu',
                           kernel_regularizer = l1(params['weight_regularizer_2'])))
    model.add(layers.Dense(2, activation = 'linear'))
    model.compile(loss = 'mse',
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['mae'])

# run the neural network model
    out = model.fit(train_data, train_labels, epochs= params['epochs'],
                    batch_size=params['batch_size'], validation_data=[test_data, test_labels], # data automatically split from 30% of total data seems like
                    verbose=0)

    return out, model

##############################################################################

# building parameter space
p = {'lr': [0.5,0.75,1,], # 0.1 -> 10 in 10 steps
     'first_neuron':[64, 32], # list of all permissible hidden nodes in first neuron
     'second_neuron': [128,64,32],
     'batch_size': [50,100], # 5 -> 100 in 20 steps
     'epochs': [200, 300], # 100 -> 500 in 5 steps
     'optimizer': [rmsprop], #  list of scanned optimizers
     #'loss': [mae], # list of scanned loss functions
    # 'activation': [relu],
     #'dropout': [0, 0.1, 0.2],
     #'hidden_layers': [1,2,3,4],
     'weight_regularizer_1':[0.0001,0.001, 0.01],
     'weight_regularizer_2':[0.0001,0.001, 0.01]
    }

##############################################################################

# run hyper parameter scan
h = ta.Scan(x = train_data, y = train_labels, params=p,
            x_val = test_data, y_val= test_labels,
            model = my_model,
            val_split= None,
            dataset_name= 'DISS',
            experiment_no='msc_wl_test',
            grid_downsample = 0.2,
            print_params= True,
            shuffle = True,
            reduce_loss = True,
            reduction_metric = 'mae')