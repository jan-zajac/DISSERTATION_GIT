# main code
import sys
sys.path.insert(0, '/Users/janzajac/PycharmProjects/YEAR 3 INDIVIDUAL PROJECT/tools')
import tools
from keras import models
from keras import layers
import numpy as np
import talos as ta
from keras.optimizers import RMSprop, Adadelta
import matplotlib.pyplot as plt
from keras.regularizers import l1
from talos.model.normalizers import lr_normalizer

##############################################################################
# read in data
data_filename = '0_p_d_kA_kE_ODE_sols_80%.txt'
label_filename = '0_p_d_kA_kE_targets_80%.txt'

# read in data
data = tools.read_data(data_filename)
labels = tools.read_data(label_filename)
##############################################################################
# splitting data into training and test
# setting ratio dictating how we are going to divide the data into training and testing
ratio = 0.8
train_data, test_data, train_labels, test_labels = tools.data_split(data,labels, ratio)
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)
##############################################################################
# standardising by the test data to make mean 0 and std = 1
# here we are normalising inputs with method 1 - where each time step can be considered as a unique feature
train_data_mean = train_data.mean(axis = 0)
train_data -= train_data_mean
test_data -= train_data_mean

train_data_std = train_data.std(axis = 0)
train_data /= train_data_std
test_data /= train_data_std

train_data[:,0] = 0 # changing all the values at t = 0 to 0s because they are nan [:,:,0] if working with 3D array
test_data[:,0] = 0 # same for test data
##############################################################################
# building of model
# anything with parameters in front of it is going to be scanned according to the parameters dictionary

def my_model(partial_train_data, partial_train_labels, val_train_data, val_train_labels, params):

    model = models.Sequential()
    model.add(layers.Dense(params['first_neuron'], activation = 'relu', input_shape=(train_data.shape[1],),
                           kernel_initializer='normal',
                           kernel_regularizer = l1(params['weight_regulizer_1'])))
    model.add(layers.Dense(params['second_neuron'], activation = 'relu'))
    model.add(layers.Dense(params['third_neuron'], activation = 'relu'))

    model.add(layers.Dense(4)) #activation = 'linear'

    model.compile(loss = 'mse',
                  optimizer = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['mae'])


# run the neural network model
    out = model.fit(partial_train_data, partial_train_labels, epochs= params['epochs'],
                    batch_size=params['batch_size'], validation_data=[val_train_data, val_train_labels], # data automatically split from 30% of total data seems like
                    verbose=0)

    return out, model

# building parameter space
p = { 'lr': (0.05, 0.5, 10),
     'first_neuron':[16, 8, 4], # list of all permissible hidden nodes in first neuron
     'second_neuron': [32, 8 ,4],
     'third_neuron': [128, 64, 32],
     'batch_size': [100,200],
     'epochs': [200],
     'optimizer': [Adadelta], #  list of scanned optimizers
     #'loss': [mae], # list of scanned loss functions
     #'activation': [relu],
     #'dropout': [0, 0.01,0.1],
     #'hidden_layers': [1,2,3,4],
     'weight_regulizer_1':[0, 0.0001, 0.001, 0.01],
     'weight_regulizer_2':[0, 0.0001, 0.001, 0.01]}

# run hyper parameter scan
h = ta.Scan(x = train_data, y = train_labels, params=p,
            x_val = test_data, y_val= test_labels,
            model = my_model,
            val_split= None,
            dataset_name= 'test',
            experiment_no='05_3_p_d_kA_kE_80%',
            grid_downsample = 0.1,
            print_params= True,
            shuffle = True,
            reduce_loss = True,
            reduction_metric = 'mae')