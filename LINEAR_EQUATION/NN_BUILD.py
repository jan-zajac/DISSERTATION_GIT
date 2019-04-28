# main code containing ANN build as well as plots for monitoring performance
# at the end code is deployed on some fabricated test data

import sys
sys.path.insert(0, '../tools')
import tools # python file containing several UDFs
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
##############################################################################

# read in data
data_filename = 'simple_ODE_sols.txt'
label_filename = 'k_vals.txt'

##############################################################################

# read in data using tools Python file
data = tools.read_data(data_filename)
labels = tools.read_labels(label_filename)

##############################################################################

# shuffling of data so that it's not in order
data, labels = tools.shuffle(data, labels)

##############################################################################

# making data and labels in array format
data_array = tools.form_array(data)
label_array = tools.form_array(labels)

##############################################################################

# splitting of data into training and test data
ratio = 0.75
train_data, test_data, train_labels, test_labels = tools.data_split(data_array,label_array, ratio)

##############################################################################

# building of model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer='rmsprop', loss='mae', metrics= [tools.coeff_determination, 'mae'])

# run the neural network model
num_epochs = 100
history = model.fit(train_data, train_labels, epochs= num_epochs,
                    batch_size=1, validation_split= 0.25)

##############################################################################

# extract history data from the model.
history_dict = history.history
loss_values = history_dict['loss'] # Train data loss values
val_loss_values = history_dict['val_loss'] # Validation data loss values
train_mae = history_dict['mean_absolute_error'] # Train data loss values
val_mae = history_dict['val_mean_absolute_error'] # Validation data loss values
train_metrics = history_dict['coeff_determination']
val_metrics = history_dict['val_coeff_determination']

##############################################################################
# creating list of epochs for plotting
epochs = range(1, len(loss_values) + 1)

# Plotting of raw training and validation data
# MSE
plt.grid()
plt.plot(epochs[10:], loss_values[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_loss_values[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION LOSS')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# R2
plt.grid()
plt.plot(epochs[10:], train_metrics[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_metrics[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION METRIC')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.pause(20)
plt.show()

# MAE
plt.grid()
plt.plot(epochs[10:], train_mae[10:], label = 'TRAINING')
plt.plot(epochs[10:], val_mae[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.pause(20) # pause so i can screen shot
plt.show()
##############################################################################

# using smoothing function in tools Python file to provide smoother plots of training history
smooth_loss = tools.smooth_curve(loss_values)
smooth_val_loss = tools.smooth_curve(val_loss_values)
smooth_train_metric = tools.smooth_curve(train_metrics)
smooth_val_metric = tools.smooth_curve(val_metrics)
smooth_train_mae = tools.smooth_curve(train_mae)
smooth_val_mae = tools.smooth_curve(val_mae)

# Plotting of smooth training and validation data
# MSE
plt.grid()
plt.plot(epochs[10:], smooth_loss[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_loss[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION LOSS')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# R2
plt.grid()
plt.plot(epochs[10:], smooth_train_metric[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_metric[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION METRIC')
plt.xlabel('Epochs')
plt.ylabel('R2 Value')
plt.legend()
plt.show()

# MAE
plt.grid()
plt.plot(epochs[10:], smooth_train_mae[10:], label = 'TRAINING')
plt.plot(epochs[10:], smooth_val_mae[10:], label = 'VALIDATION')
plt.title('TRAINING VS VALIDATION MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

##############################################################################

# evaluating the model based on test data
# evaluate the model
test_mse, test_metric, test_mae = model.evaluate(test_data, test_labels)
print('THE TEST LOSS IS ' + str(test_mse))
print('THE TEST METRIC IS ' + str(test_metric))
print('THE TEST MAE IS ' + str(test_mae))

##############################################################################

# testing model on a set of data for which value of k = 1500
print('TESTING NEURAL NETWORK on K = 1500!')
prediction_data = [0, 15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000, 150000]
prediction_data = np.asarray(prediction_data)
prediction_data = prediction_data.reshape(-1,11)

result = model.predict(prediction_data)
print('RESULT IS ' + str(result))

# testing model on a set of data for which value of k = -1500
print('TESTING NEURAL NETWORK on K = -1500!')
prediction_data = [0, -15000, -30000, -45000, -60000, -75000, -90000, -105000, -120000, -135000, -150000]
prediction_data = np.asarray(prediction_data)
prediction_data = prediction_data.reshape(-1,11)

result = model.predict(prediction_data)
print('RESULT IS ' + str(result))










