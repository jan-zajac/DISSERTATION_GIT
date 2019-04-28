# code to generate the data for my simple example
# the function in question = y = kt
# so dy/dt = k
# will solve ODE for values of k ranging from -1000 - 1000 (2000  samples)
# time vector will be identical for all datasets so will generate it, but not include in building the NN
# the time vector will have length 11, spanning from 0-100 in 10s time steps

import numpy as np
from scipy.integrate import odeint #shortens scipy.integrate.odeint to just odeint

# the number of time points we want to have
no_timepoints = 11

# the final last time point we want to solve until
final_time = 100

# the time vector we are going to use for every solution of the ODE
t = np.linspace(0,final_time, num = no_timepoints)

# creating list of k values from 0-999
k = list(range(-1000,1000))

# definition of ODE
def model(y,t, k):
    dydt = k
    return dydt

#initial condition
y0 = 0

# creating an empty matrix of zeros that we are going to fill in as we solve all the ODEs with different k values
dependent_var = np.zeros((len(k), no_timepoints))

# for loop solivng ODEs for each k value in the list
for i in range(len(k)):
    # solving the ODE with odeint
    y = odeint(model, y0, t, args = (k[i],))

    # need to transpose dependent variable  to update the new matrix.
    y = np.transpose(y, axes=None)

    # adding solved ODE to array of solutions
    dependent_var[i] = y

# converting the dependent var data into integers to make easy reading in data later
dependent_var = dependent_var.astype(int)

#saving input and target data as
#np.savetxt('simple_ODE_sols.txt', dependent_var, fmt = '%i', delimiter= ',' )
#np.savetxt('k_vals.txt', k, fmt = '%i', delimiter= ',' )