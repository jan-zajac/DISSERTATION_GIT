# main code building the optimised ANN

import sys
sys.path.insert(0, '../../tools')
import tools
from keras import models
from keras import layers
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt

##############################################################################
# read in data
data_filename = '0_p_d_kA_kE_ODE_sols.txt'
label_filename = '0_p_d_kA_kE_targets.txt'

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
num_epochs = 300
batch_size = 50

# building of model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],),
                           kernel_initializer='normal'))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(4, activation = 'linear'))

rmsprop = optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop, loss='mse', metrics=[tools.coeff_determination, 'mae'])

# run the neural network model
history = model.fit(train_data, train_labels, epochs= num_epochs,
                        batch_size = batch_size, validation_data=(test_data, test_labels))
##############################################################################

# extract history data from the model.
history_dict = history.history
loss_values = history_dict['loss'] # Train data loss values
val_loss_values = history_dict['val_loss'] # Validation data loss values
train_mae = history_dict['mean_absolute_error']  # Train data MAE values
val_mae = history_dict['val_mean_absolute_error']  # Validation data MAE values
train_r2 = history_dict['coeff_determination'] # Train data R2 VALUES
val_r2 = history_dict['val_coeff_determination'] # Validation data R2 values
epochs = range(1, len(loss_values) + 1)

##############################################################################

# Plotting of raw training and validation data
# MSE
plt.plot(epochs[10:], loss_values[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_loss_values[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION LOSS')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# R2
plt.plot(epochs[10:], train_r2[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_r2[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION MAE')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.show()

# MAE
plt.plot(epochs[10:], train_mae[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_mae[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

##############################################################################

# using smoothing function in tools Python file to provide smoother plots of training history
smooth_loss = tools.smooth_curve(loss_values)
smooth_val_loss = tools.smooth_curve(val_loss_values)
smooth_train_r2 = tools.smooth_curve(train_r2)
smooth_val_r2 = tools.smooth_curve(val_r2)
smooth_train_mae = tools.smooth_curve(train_mae)
smooth_val_mae = tools.smooth_curve(val_mae)

# Plotting of raw training and validation data
# MSE
plt.plot(epochs[10:], smooth_loss[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_loss[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION LOSS')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', labelsize=18)
plt.grid()
plt.show()

# R2
plt.plot(epochs[10:], smooth_train_r2[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_r2[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', labelsize=18)
plt.grid()
plt.show()

# MAE
plt.plot(epochs[10:], smooth_train_mae[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_mae[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', labelsize=18)
plt.grid()
plt.show()

##############################################################################

# evaluate the model
test_mse_score, test_r2_score, test_mae_score = model.evaluate(test_data, test_labels)
print('RESULTS FOR TEST DATA ARE:')
print('THE TEST LOSS IS: ' + str(test_mse_score))
print(' THE TEST R2 IS: ' + str(test_r2_score))
print(' THE TEST MAE IS: ' + str(test_mae_score))

##############################################################################

# serialize model to JSON
model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")