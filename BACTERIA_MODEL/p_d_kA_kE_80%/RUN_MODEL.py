# Running model from file and comparing output trajectories

import sys
sys.path.insert(0, '../../tools')
import tools
import numpy as np
from keras.models import model_from_json
from random import randint
import matplotlib.pyplot as plt
from scipy.integrate import odeint #shortens scipy.integrate.odeint to just odeint
from statistics import mean

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
# now read in prediction data
data_filename = '0_p_d_kA_kE_ODE_sols_TEST_2000_80%.txt'
label_filename = '0_p_d_kA_kE_targets_TEST_2000_80%.txt'
prediction_data = tools.read_data(data_filename)
prediction_labels = tools.read_data(label_filename)
orig_pred_data = prediction_data
orig_pred_labels = prediction_labels

prediction_data = np.asarray(prediction_data)
prediction_labels = np.asarray(prediction_labels)
##############################################################################
# standardizing prediction data and target
prediction_data -= train_data_mean
prediction_data /= train_data_std
prediction_data[:,0] = 0 # changing all the values at t = 0 to 0s because they are nan
##############################################################################
# read in model in JSON format
json_file = open('talos_exp_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
##############################################################################
# load weights into new model
loaded_model.load_weights('talos_exp_1.h5')
print('loaded model')
#compile model
loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=[tools.coeff_determination,'mae']) # tools.coeff_determination
##############################################################################
# running the model on testing data
val_loss, val_metric, val_mae = loaded_model.evaluate(prediction_data, prediction_labels, verbose=0)
print('the test loss is ' + str(val_loss))
print('the test metric is ' + str(val_metric))
print('the test mae is ' + str(val_mae))
##############################################################################

# print results
print('TESTING NEURAL NETWORK on standardised  = ' + str(prediction_labels[0])) # testing on first sample in prediction data ste
prediction_data_vec = np.asarray(prediction_data[0])
prediction_data_vec = prediction_data_vec.reshape(-1,100)
result = loaded_model.predict(prediction_data_vec)
print('STANDARDIZED RESULT IS ' + str(result))

##############################################################################
# hardcoding constants with values from figure 2
VR = 2.0
VA = 2.0
KR = 1.0
KA = 1.0
R0 = 0.05
A0 = 0.05
kR = 0.7
# defining the ODE to be integrated in this function
def model(Z, t):
    # Z a list of solutions for R and A
    R = Z[0]
    A = Z[1]

    # defining the ODE for R - according to equation (20) in the 2001 paper
    dRdt = (-kR * R) + ((VR * R * A) / (KR + (R * A))) + R0

    # defining the equation for decay rate - according to equation (21) in the 2001 paper
    dp = kA + ((d / p) * ((kE * (1 - p)) / (d + (kE * (1 - p)))))

    # defining the ODE for A - according to equation (21) in the 2001 paper
    dAdt = ((VA * R * A) / (KA + (R * A))) + A0 - (dp * A)

    # create a list of the 2 equations, 1st one is R, second is A
    dZdt = [dRdt, dAdt]
    return dZdt
##############################################################################
# plotting some of the result real trajectories to those outputted by the constants predicted by NN
# initial condition

Z0 = [0,0]

# defining time period
fin_time = 40
time_step = 100 # also defines the number of time points each data set will have
t_long = np.linspace(0,fin_time, time_step)
##############################################################################

# for loop that plots some of the sample of prediction data and resultant model data
for i in range(100):
    # assigning random index
    index = randint(0,len(prediction_data))

    # plotting the original real data
    plt.plot(t_long, orig_pred_data[index], linewidth=3, linestyle=':', label='ORIGINAL TRAJECTORY')
    plt.title('Comparison of real vs model predicted trajectories')
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.grid()

    # solving the constants by inputting the data
    # Miscellaneous
    print('TESTING NEURAL NETWORK on standardised again = ' + str(prediction_labels[index]))
    prediction_data_vec = np.asarray(prediction_data[index])
    prediction_data_vec = prediction_data_vec.reshape(-1, 100)
    result = loaded_model.predict(prediction_data_vec)
    print('STANDARDIZED RESULT IS again' + str(result))
    d = result[0][0]# Diffusion conductance through cell membrane
    p = result[0][1]  # Local density (volume fraction) of cell
    kA = result[0][2]
    kE = result[0][3]


    # defining the ODE to be integrated in this function
    def model(Z, t):
        # Z a list of solutions for R and A
        R = Z[0]
        A = Z[1]

        # defining the ODE for R - according to equation (20) in the 2001 paper
        dRdt = (-kR * R) + ((VR * R * A) / (KR + (R * A))) + R0

        # defining the equation for decay rate - according to equation (21) in the 2001 paper
        dp = kA + ((d / p) * ((kE * (1 - p)) / (d + (kE * (1 - p)))))

        # defining the ODE for A - according to equation (21) in the 2001 paper
        dAdt = ((VA * R * A) / (KA + (R * A))) + A0 - (dp * A)

        # create a list of the 2 equations, 1st one is R, second is A
        dZdt = [dRdt, dAdt]
        return dZdt

    # solving the ode
    Z = odeint(model, Z0, t_long)
    Z = np.transpose(Z, axes=None)  # comment out if I want to plot!!!

    # appending solutions FOR PLOTTING ONLY
    R = Z[0]
    A = Z[1]

    # plotting the ODEs to check behaviour
    # plots against time
    plt.plot(t_long,A, linewidth = 3, linestyle = '-',label = 'PREDICTED TRAJECTORY')
    plt.xlabel('time')
    plt.legend(loc='best', prop={'size': 14})
    plt.ylabel('concentration')
    plt.tick_params(axis = 'both', labelsize = 18)

    plt.show()

##############################################################################

# section for plotting mean error of the constants predicted by the neural network
# initialising empty target lists
pred_d_list = []
pred_p_list = []
pred_kA_list = []
pred_kE_list = []

for i,j in enumerate(orig_pred_data):
    print('TESTING NEURAL NETWORK on standardised again = ' + str(prediction_labels[i]))
    prediction_data_vec = np.asarray(prediction_data[i])
    prediction_data_vec = prediction_data_vec.reshape(-1, 100)
    result = loaded_model.predict(prediction_data_vec)
    print('STANDARDIZED RESULT IS again' + str(result))
    d = result[0][0]  # Diffusion conductance through cell membrane
    p = result[0][1]  # Local density (volume fraction) of cell
    kA = result[0][2]
    kE = result[0][3]

    pred_d_list.append(d)
    pred_p_list.append(p)
    pred_kA_list.append(kA)
    pred_kE_list.append(kE)

# calculating the errors for the different coefficients
err_d = []
err_p = []
err_kA = []
err_kE = []

# calculating the percentage error
for i,j in enumerate(orig_pred_labels):
    err_d.append((abs(j[0] - pred_d_list[i])/abs(j[0])*100))
    err_p.append((abs(j[1] - pred_p_list[i])/abs(j[1])*100))
    err_kA.append((abs(j[2] - pred_kA_list[i])/abs(j[2])*100))
    err_kE.append((abs(j[3] - pred_kE_list[i])/abs(j[3])*100))

d_mean = mean(err_d)
p_mean = mean(err_p)
kA_mean = mean(err_kA)
kE_mean = mean(err_kE)
mean_scores = [d_mean, p_mean, kA_mean, kE_mean]
const_list = ['d', 'p', 'kA', 'kE']
x_pos = [i for i, _ in enumerate(const_list)]

# plotting bar chart of the different constant means from the test data
fig,ax = plt.subplots(1)
ax.bar(const_list, mean_scores)
ax.set_title('MEAN CONSTANT ERROR')
ax.tick_params(axis = 'both', labelsize = 18)
ax.set_xticklabels([]) # hiding the x axis labels
plt.show()

##############################################################################
# for loop that will iterate through all the prediction data and calculate the mean absolute error nad r2?
# by checking how different each time point is from the concentration
sols = []
for i, j in enumerate(orig_pred_labels):

    # solving the constants by inputting the data
    # Miscellaneous
    print('TESTING NEURAL NETWORK on standardised again = ' + str(prediction_labels[i]))
    prediction_data_vec = np.asarray(prediction_data[i])
    prediction_data_vec = prediction_data_vec.reshape(-1, 100)
    result = loaded_model.predict(prediction_data_vec)
    print('STANDARDIZED RESULT IS again' + str(result))
    d = result[0][0]  # Diffusion conductance through cell membrane
    p = result[0][1]  # Local density (volume fraction) of cell
    kA = result[0][2]
    kE = result[0][3]

    # solving the ode
    Z = odeint(model, Z0, t_long)
    Z = np.transpose(Z, axes=None)  # comment out if I want to plot!!!

    # appending solutions FOR PLOTTING ONLY
    R = Z[0]
    A = Z[1]
    sols.append(A)
##############################################################################
# calculating the error at each time point
ers_all = []
for i, j in zip(orig_pred_data,sols):
    ers_ind = [] # empty list to calculate each error
    for x, y in zip(i, j):
        ers_ind.append(abs(x-y))
    ers_all.append(ers_ind)

ers_mean = []
for i in ers_all:
    ers_mean.append(mean(i))
total_er = mean(ers_mean)
print('THE AVERAGE ABSOLUTE ERROR OF ALL THE AVERAGE ERRORS AT EACH TIME POINT IS ' + str(total_er))