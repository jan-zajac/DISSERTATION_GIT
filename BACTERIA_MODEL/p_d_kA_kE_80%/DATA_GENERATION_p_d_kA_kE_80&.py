# data generation code that varies only in terms of p, d, kE and kA and has four targets by 80%

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint #shortens scipy.integrate.odeint to just odeint
import seaborn as sns

##############################################################################
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
# hardcoding constants with values from literature
VR = 2.0
VA = 2.0
KR = 1.0
KA = 1.0
R0 = 0.05
A0 = 0.05
kR = 0.7
##############################################################################
#initial condition
# [R0,A0]
Z0 = [0,0]

# defining time period
fin_time = 40
time_step = 100 # also defines the number of time points each data set will have
t = np.linspace(0,fin_time, time_step)

# empty lists for R and A FOR PLOTTING
R_plot = []
A_plot = []

# defining how many datasets I want to produce
tot_data = 10000
##############################################################################
# initialising empty target lists
d_list = []
p_list = []
kA_list = []
kE_list = []

const_list_str = ['d', 'p', 'kA', 'kE']
##############################################################################
# empty list for analysing differences between initial and final concentrations
diff_R = []
diff_A = []
#empty containers fot storing the actual ODE solutions
R_sols = []
A_sols = []

# container for solutions when trying to do 3D array data (R and A in the same container)
sols = []
for i in range(tot_data):
    # Miscellaneous
    d = round(random.uniform(0.04, 0.36), 4)  # Diffusion conductance through cell membrane
    p = round(random.uniform(0.02, 0.18), 4)  # Local density (volume fraction) of cell
    kA = round(random.uniform(0.004, 0.036), 4) # Natural degradation rate of intracellular 3-oxo-C12-HSL (autoinducer)
    kE = round(random.uniform(0.02, 0.18), 4) # Natural degradation rate of extracellular 3-oxo-C12-HSL (autoinducer)

    const_list_float = [d, p]
    # appending list for targets
    d_list.append(d)
    p_list.append(p)
    kA_list.append(kA)
    kE_list.append(kE)

    # solving the ode
    Z = odeint(model, Z0, t)
    Z = np.transpose(Z, axes=None) # comment out if I want to plot!!!

    sols.append(Z) # this appends both of the solved ODEs into what would be a 3D array of shape (1000, 2, 100)

    # appending solutions FOR PLOTTING ONLY
    R = Z[0]
    A = Z[1]

    # appending empty list with ODE solutions
    R_sols.append(R)
    A_sols.append(A)

    # appending the differences between the final and initial values
    diff_R.append(abs(R[0]-R[-1]))
    diff_A.append(abs(A[0]-A[-1]))

    # plotting the ODEs to check behaviour
    # COMMENT THIS OUT TO COMPLETE CODE
    plt.plot(t, Z[0], linewidth=2, linestyle='-', label='R')
    plt.plot(t, Z[1], linewidth=2, linestyle=':', label='A')
    plt.title('TOXIN CONCENTRATION OVER TIME')
    plt.xlabel('TIME')
    plt.legend(loc='best')
    plt.ylabel('CONCENTRATION')
    plt.grid()

    plt.show()

##############################################################################

# plotting the distribution of distances
sns.distplot(diff_R, hist=True, rug=False, label='R')
sns.distplot(diff_A, hist=True, rug=False, label='A')
plt.xlabel('difference between concentrations')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.show()

# plotting the scatter of concentration differences
plt.scatter(range(len(diff_R)), diff_R, label='R', s=2, marker='x')
plt.scatter(range(len(diff_A)), diff_A, label='A', s=2, marker='.')
plt.xlabel('sample number')
plt.ylabel('absolute difference betwen final and initial values')
plt.legend(loc='best')
plt.grid()
plt.show()

##############################################################################

# seeing how many of the differences are greater than 1 to define QS switch
R_switch = sum(i > 1 for i in diff_R)
A_switch = sum(i > 1 for i in diff_A)

print('total of R QS switches = ' + str(R_switch))
print('total of A QS switches = ' + str(A_switch))

##############################################################################

# stacking and saving the input and target arrays.
targets = np.column_stack((d_list, p_list))
targets = np.column_stack((targets, kA_list))
targets = np.column_stack((targets, kE_list))
sols = np.asarray(sols)
A_sols = np.asarray(A_sols)

# save data to files
# np.savetxt('0_p_d_kA_kE_ODE_sols_80%.txt', A_sols, delimiter= ',' )
# np.savetxt('0_p_d_kA_kE_targets_80%.txt.txt', targets, delimiter= ',')