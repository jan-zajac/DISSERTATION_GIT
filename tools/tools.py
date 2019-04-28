# python script with  UDFs that will clean up maing code for building the network
import random
import numpy as np
from keras import backend as K

# function that reads in data outputted by my data generation script
def read_data(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        data = []
        # the following for loop appends each line of data to data list, stripping away all the unnecessary strings
        for line in lines:
            line = line.rstrip('\n')
            line = line.split(',')
            data.append(line)
    data = [[float(j) for j in i] for i in data]
    return data

# different function for reading label data
def read_labels(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        k_vals = []
        for line in lines:
            k_vals.append(float(line))
    return k_vals

# function that shuffles equally the ODE data and labels
def shuffle(data, labels):
    c = list(zip(data, labels))
    random.shuffle(c)
    data, labels = zip(*c)
    return data, labels

# function that will return data in array format
def form_array(data):
    data = np.array(data)
    return data

# function that will split data into training/test data
def data_split(data, labels, ratio):
    train_data = data[:int(len(data)*ratio)]
    test_data = data[int(len(data)*ratio):]
    train_labels = labels[:int(len(labels)*ratio)]
    test_labels = labels[int(len(labels)*ratio):]
    return train_data, test_data, train_labels, test_labels

# function that saves history data of training and validation losses and metrics
def save_history_data(epochs, loss_values, val_loss_values, metrics_values, val_metrics_values):
    loss_values_vs_epochs = np.column_stack((epochs, loss_values, val_loss_values))
    mae_values_vs_epochs = np.column_stack((epochs, metrics_values, val_metrics_values))
    np.savetxt(input('how do you want to call loss vs epochs file ? ') + '.txt', loss_values_vs_epochs, delimiter=',')
    np.savetxt(input('how do you want to call mae_loss vs epochs file ? ') + '.txt', mae_values_vs_epochs,
               delimiter=',')

# function from Chollet PDF that smooths out history data
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

# function for calculating R2 value to be used as metric.
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def plot_max_diff(A):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # empty list for analysing differences between initial and final concentrations
    diff_A = []
    for i in A:
        diff_A.append(abs(i[0] - i[-1]))

    ##############################################################################

    # plotting the distribution of distances
    sns.distplot(diff_A, hist=True, rug=False, label='A')
    plt.xlabel('difference between concentrations')
    plt.ylabel('probability density')
    plt.legend(loc='best')
    plt.show()

    plt.scatter(range(len(diff_A)), diff_A, label='A', s=2, marker='.')
    plt.xlabel('sample number')
    plt.ylabel('absolute difference betwen final and initial values')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    ##############################################################################

    # seeing how many of the differences are greater than 1 to define QS switch
    A_switch = sum(i > 1 for i in diff_A)
    print('total of A QS switches = ' + str(A_switch))

