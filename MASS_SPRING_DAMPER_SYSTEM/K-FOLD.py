# main code for carrying ou K-FOLD cross validation for mass/spring/damper system

import sys
sys.path.insert(0, '/Users/janzajac/PycharmProjects/YEAR 3 INDIVIDUAL PROJECT/tools')
import tools
from keras import models
from keras import layers
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
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

# operation on test data
train_data_std = train_data.std(axis = 0)
train_data /= train_data_std
test_data /= train_data_std

# changing all the values at t = 0 to 0s because they are nan
train_data[:,0] = 0
test_data[:,0] = 0

##############################################################################

# standardising target labels in same way
target_mean = train_labels.mean(axis = 0)
train_labels -= target_mean
test_labels -= target_mean

target_std = train_labels.std(axis = 0)
train_labels /= target_std
test_labels /= target_std

##############################################################################

# doing a k-fold validation
k = 4 # number of folds
num_val_samples = len(train_data) // k # calculating number of data samples for each fold

# initialising history data lists
num_epochs = 300
all_mae_scores = []
all_r2_scores = []
all_mae_histories = []
all_r2_histories = []

# for loop for k-fold cross validation
for i in range(k):
    print('processing fold #', i)
    val_train_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] # assigning validation data by slicing data based on fold iteration
    val_train_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples] # same for labels
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],  # creating the partial training data
                                          train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_labels = np.concatenate([train_labels[:i * num_val_samples], # creating the partial training labels
                                            train_labels[(i + 1) * num_val_samples:]], axis=0)

# building of model
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],),
                           kernel_initializer='normal', kernel_regularizer = l1(0.0001)))
    model.add(layers.Dense(128, activation = 'relu', kernel_regularizer = l1(0.0001)))
    model.add(layers.Dense(2, activation = 'linear'))

    rmsprop = optimizers.RMSprop(lr = 0.005)

    model.compile(optimizer=rmsprop, loss='mse', metrics=[tools.coeff_determination, 'mae']) #tools.coeff_determination]


# run the neural network model
    history = model.fit(partial_train_data, partial_train_labels, epochs= num_epochs,
                        batch_size=50, validation_data=(val_train_data, val_train_labels))

    # evaluating the model based on validation data
    val_mse, val_r2,val_mae = model.evaluate(val_train_data, val_train_labels, verbose=0)

    all_r2_scores.append(val_r2)  # appending fold validation R2 score
    all_mae_scores.append(val_mae) # appending fold validation MAE score

    mae_score_mean = np.mean(all_mae_scores) # calculating mean of MAE scores
    r2_score_mean = np.mean(all_r2_scores) # calculating mean of R2 scores

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

    # appending mae history
    all_mae_histories.append(train_mae)
    all_r2_histories.append(train_r2)

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
    plt.pause(20) # pause so i can screen shot
    plt.show()

    # MAE
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

# calculating average of final validation metric for each fold
mae_score_mean = np.mean(all_mae_scores)
r2_score_mean = np.mean(all_r2_scores)
print('THE VALIDATION MAE SCORES FOR EACH FOLD ARE: ')
print(all_mae_scores)

print('THE VALIDATION R2 SCORES FOR EACH FOLD ARE: ')
print(all_r2_scores)

# adding the mean to the total scores
all_mae_scores.append(mae_score_mean)
all_r2_scores.append(r2_score_mean)
##############################################################################

# calculating the average metric history from the histories of each fold
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

average_r2_history = [
    np.mean([x[i] for x in all_r2_histories]) for i in range(num_epochs)]

##############################################################################

# Plotting bar chart for average scores of each fold
# formatting x axis for plt.bar function
i_range = list(range(k))
i_range.append('mean')
i_range = list(map(str, i_range))
x_pos = [i for i, _ in enumerate(i_range)]

# plotting bar chart of the different validation MAE means from k-fold validation
plt.bar(x_pos, all_mae_scores)
plt.title('VALIDATIONS METRICS FOR EACH FOLD')
plt.xlabel('ITERATION')
plt.xticks(x_pos, i_range)
plt.tick_params(axis='both', labelsize=18)
plt.ylabel('VALIDATION MAE')
plt.grid()
plt.show()


# plotting bar chart of the different validation R2 means from k-fold validation
plt.bar(x_pos, all_r2_scores)
plt.title('VALIDATIONS METRICS FOR EACH FOLD')
plt.xlabel('ITERATION')
plt.xticks(x_pos, i_range)
plt.tick_params(axis='both', labelsize=18)
plt.ylabel('VALIDATION R2')
plt.grid()
plt.show()