# main code
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
data_filename = '0_p_d_kA_kE_ODE_sols_80%.txt'
label_filename = '0_p_d_kA_kE_targets_80%.txt'

# read in data
data = tools.read_data(data_filename)
labels = tools.read_data(label_filename)
##############################################################################
"""
# downscaling my input data to be 10 time steps rather than 100
new_data = []
for i in data:
    new_data.append(i[0::10])
data = new_data
"""
##############################################################################
"""
# plotting the ODEs to check behaviour
#plots against time
# defining time period
fin_time = 40
time_step = 10 # also defines the number of time points each data set will have
t = np.linspace(0,fin_time, time_step)
for i in range(100):
    plt.plot(t,data[i], linewidth = 1, linestyle = ':',label = 'A')
    plt.title('INTEGRATED ODEs for R and A')
    plt.xlabel('time')
    #plt.legend(loc='best')
    plt.ylabel('concentration')
    plt.grid()

plt.show()
sys.exit()
"""
##############################################################################
"""
# reducing the labels to only one value (going to try p)
new_labels = []
for i in labels:
    new_labels.append(i[1])
labels = new_labels
"""
##############################################################################
"""
# reducing dataset size to see if I can see overfitting
data = data[:100]
labels = labels[:100]
"""
##############################################################################
"""
# normalising outputs using method 3
labels = np.asarray(labels)
labels = labels.transpose() # transpose so that when we loop, we are considering all the data at the respective time step
norm_labels= np.zeros(labels.shape) # creating container for normalized data
for i,j in enumerate(labels):
    values = j.reshape((len(j), 1)) # reshaping into single column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    normalized = scaler.transform(values) # normalization happens here
    normalized = normalized.reshape((1, len(j))) # reshaping back into single row
    norm_labels[i] = normalized

norm_labels = norm_labels.transpose() # need to transpose again at the end
labels = norm_labels
"""
##############################################################################
# splitting data into training and test
# setting ratio dictating how we are going to divide the data into training and testing
ratio = 0.8
train_data, test_data, train_labels, test_labels = tools.data_split(data,labels, ratio)
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

# plot
#fig = plt.figure(figsize=(15, 5))
#fig.subplots_adjust(wspace=0.5, hspace=0.3)
#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)
#ax1.scatter(train_labels[:,0],train_labels[:,1])
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
"""
# here we are normalising the targets by method 6

target_mean = train_labels.mean(axis = 0)
train_labels -= target_mean
test_labels -= target_mean

target_std = train_labels.std(axis = 0)
train_labels /= target_std
test_labels /= target_std

#ax2.scatter(train_labels[:,0],train_labels[:,1])
#plt.show()
"""
##############################################################################
"""
# plotting a correlation heatmap for all the timepoints to check collinearity FOR TRAIN DATA
data_corr = np.corrcoef(train_data.T)

fig, ax = plt.subplots()
# plotting the correlation heatmap
sns.heatmap(data_corr,
        xticklabels=list(range(100)),
        yticklabels=list(range(100)))

# chanig the tick font size
plt.tick_params(axis = 'both', labelsize = 6)

# making every 5th timepoint visible as a tick
#plt.xticks(np.arange(min(list(range(100))), max(list(range(100)))+1, 1.0))
loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

plt.show()
"""
##############################################################################
# doing a k-fold validation
k = 4 # number of folds
num_val_samples = len(train_data) // k # calculating number of data samples for each fold
num_epochs = 300
batch_size = 200

all_metric_scores = []
all_mae_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_train_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] # assigning validation data by
                                                                          # slicing data based on fold iteration
    val_train_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples] # same for labels
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],  # creating the partial training data
                                          train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_labels = np.concatenate([train_labels[:i * num_val_samples], # creating the partial training labels
                                            train_labels[(i + 1) * num_val_samples:]], axis=0)

# building of model
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],),
                           kernel_initializer='he_uniform')) #kernel_regularizer = l1(0.0001)
    model.add(layers.Dense(64, activation = 'relu')) #kernel_regularizer = l1(0.0001)
    model.add(layers.Dense(4, activation = 'linear'))

    rmsprop = optimizers.RMSprop(lr = 0.001)

    model.compile(optimizer=rmsprop, loss='mse', metrics=[tools.coeff_determination, 'mae']) #tools.coeff_determination]


# run the neural network model
# num_epochs = 200
    history = model.fit(partial_train_data, partial_train_labels, epochs= num_epochs,
                        batch_size=batch_size, validation_data=(val_train_data, val_train_labels))

    # evaluating the model based on validation data
    val_mse, val_metric, val_mae  = model.evaluate(val_train_data, val_train_labels, verbose=0)

    all_metric_scores.append(val_metric) # appending fold validation metric score
    all_mae_scores.append(val_mae) # appending fold validation metric score

    metric_score_mean = np.mean(all_metric_scores)
    mae_score_mean = np.mean(all_mae_scores)
##############################################################################

# extract history data from the model.
    history_dict = history.history
    loss_values = history_dict['loss'] # Train data loss values
    val_loss_values = history_dict['val_loss'] # Validation data loss values
    train_mae = history_dict['mean_absolute_error']  # Train data loss values
    val_mae = history_dict['val_mean_absolute_error']  # Validation data loss values
    train_metrics = history_dict['coeff_determination']
    val_metrics = history_dict['val_coeff_determination']
    epochs = range(1, len(loss_values) + 1)

    # appending mae history
    all_mae_histories.append(train_mae)
##############################################################################
# plotting of training and validation data

    plt.plot(epochs[10:], loss_values[10:], label = 'training_loss')
    plt.plot(epochs[10:], val_loss_values[10:], label = 'validation_loss')
    plt.title('TRAINING VS VALIDATION LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.plot(epochs[10:], train_metrics[10:], label = 'training_metric_mae')
    plt.plot(epochs[10:], val_metrics[10:], label = 'validation_metric_mae')
    plt.title('TRAINING VS VALIDATION METRIC')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.pause(20) # pause so i can screen shot
    plt.show()

##############################################################################
    smooth_loss = tools.smooth_curve(loss_values)
    smooth_val_loss = tools.smooth_curve(val_loss_values)
    smooth_train_metric = tools.smooth_curve(train_metrics)
    smooth_val_metric = tools.smooth_curve(val_metrics)
    smooth_train_mae = tools.smooth_curve(train_mae)
    smooth_val_mae = tools.smooth_curve(val_mae)


    plt.plot(epochs[10:], smooth_loss[10:], label = 'smooth_training_loss')
    plt.plot(epochs[10:], smooth_val_loss[10:], label = 'smooth_validation_loss')
    plt.title('TRAINING VS VALIDATION LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.plot(epochs[10:], smooth_train_metric[10:], label = 'smooth_training_metric')
    plt.plot(epochs[10:], smooth_val_metric[10:], label = 'smooth_validation_metric')
    plt.title('TRAINING VS VALIDATION METRIC')
    plt.xlabel('Epochs')
    plt.ylabel('METRIC')
    plt.legend()
    plt.pause(20) # pause so i can screen shot
    plt.show()

    plt.plot(epochs[10:], smooth_train_mae[10:], label='smooth_training_mae')
    plt.plot(epochs[10:], smooth_val_mae[10:], label='smooth_validation_mae')
    plt.title('TRAINING VS VALIDATION MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.pause(20)  # pause so i can screen shot
    plt.show()

##############################################################################
# save history data to a text file
#tools.save_history_data(epochs, loss_values, val_loss_values, mae_values, val_mae_values)
#tools.save_history_data(epochs, loss_values, val_loss_values, r2_values, val_r2_values)

##############################################################################
# calculating average of final validation metric for each fold
metric_score_mean = np.mean(all_metric_scores)
print('THE VALIDATION METRIC SCORES FOE EACH FOLD ARE: ')
print(all_metric_scores)

# adding the mean to the total scores
all_metric_scores.append(metric_score_mean)

print('THE MEAN METRIC VALIDATION SCORE IS:')
print(metric_score_mean)
# calculating the average metric history from the histories of each fold
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# formatting x axis for plt.bar function
i_range = list(range(k))
i_range.append('mean')
i_range = list(map(str, i_range))
x_pos = [i for i, _ in enumerate(i_range)]

# plotting bar chart of the different validation means from k-fold validation
plt.bar(x_pos, all_metric_scores)
plt.title('VALIDATIONS METRICS FOR EACH FOLD')
plt.xlabel('ITERATION')
plt.xticks(x_pos, i_range)
plt.ylabel('VALIDATION METRIC')
plt.show()
##############################################################################
# calculating average of final validation MAE for each fold
mae_score_mean = np.mean(all_mae_scores)
print('THE VALIDATION MAE SCORES FOE EACH FOLD ARE: ')
print(all_mae_scores)

# adding the mean to the total scores
all_mae_scores.append(mae_score_mean)

print('THE MEAN MAE VALIDATION SCORE IS:')
print(mae_score_mean)
# calculating the average metric history from the histories of each fold
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# formatting x axis for plt.bar function
i_range = list(range(k))
i_range.append('mean')
i_range = list(map(str, i_range))
x_pos = [i for i, _ in enumerate(i_range)]

# plotting bar chart of the different validation means from k-fold validation
plt.bar(x_pos, all_mae_scores)
plt.title('VALIDATIONS MAE FOR EACH FOLD')
plt.xlabel('ITERATION')
plt.xticks(x_pos, i_range)
plt.ylabel('VALIDATION MAE')
plt.show()