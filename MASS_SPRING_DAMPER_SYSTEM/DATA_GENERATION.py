# code to generate the data for mass/spring/damper example
# will solve ODE for values of m, k and c, but target outputs will be wn and zeta
# time vector will be identical for all datasets so will generate it, but not include in building the NN
# Dataset information
# Total t = 50, collected at 2Hz
# 10,000 samples
# 1≤𝑚≤10
# 1≤𝑘≤10
# 1≤𝑐≤10
# Discard any 𝜉>10

import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint #shortens scipy.integrate.odeint to just odeint
import seaborn as sns

##################################################################################

# defining sping/mass/damper ODE
def model(y, t, m, k, c):
    ode = [y[1], -k/m*y[0] -c/m*y[1]]
    return ode

# defining a function to calculate zeta
def zeta_calc(m, k, c):
    zeta = c/(2*sqrt(k*m))
    return zeta

def wn_calc(m, k):
    wn = sqrt(k / m)
    return wn

##################################################################################

# the number of time points we want to have
no_timepoints = 100

# the final last time point we want to solve until
final_time = 50

# the time vector we are going to use for every solution of the ODE
t = np.linspace(0,final_time, num = no_timepoints)

# initial conditions
y10 = 2 # initial displacement
y20 = 3 # initial velocity

y0 = [y10, y20]

# only care about displacements so only need to store the first column of solved ODE output
displacement = []

##################################################################################

# initializing sectioning variables
m_list = []
k_list = []
c_list = []
z_list = []
wn_list = []
under_damped = []
over_damped = []
v_under_damped = []

# use this to define how many data sets I want
len_total_data = 10000
##################################################################################

# not constraining anything and calculating wn and zeta as targets
while len(m_list) != len_total_data:
    m = round(random.uniform(0.1, 10), 2)
    k = round(random.uniform(0.1, 10), 2)
    c = round(random.uniform(0.1, 10), 2)
    z = zeta_calc(m, k, c)
    wn = wn_calc(m, k)
    if z < 10:
        m_list.append(m)
        k_list.append(k)
        c_list.append(c)
        v_under_damped.append(z)
        z_list.append(z)
        wn_list.append(wn)

##############################################################################

# identifying the proportion of different damping behaviours in the data
under_damped = []
over_damped = []
v_under_damped = []

# based on calculated zeta value, add to list of corresponding behaviour
for i in z_list:
    if i <= 0.1:
        v_under_damped.append(i)
    elif i > 0.1 and i < 1:
        under_damped.append(i)
    elif i > 1:
        over_damped.append(i)

# included the equal to sign for some cases as in some, you get absolute values of 0.1 or 1...
print('the number of under damped cases is ' + str(len(under_damped)))
print('the number of over damped cases is ' + str(len(over_damped)))
print('the number of very under damped cases is ' + str(len(v_under_damped)))
total_cases = len(v_under_damped) + len(under_damped) + len(over_damped)

##############################################################################

# Plotting pie chart of ratio of damping ratio cases
z_ratio = [len(under_damped), len(over_damped), len(v_under_damped)]
labels = ['under damped', 'over damped', 'v_underly damped']

colors = ['yellowgreen', 'lightcoral', 'lightskyblue']

plt.pie(z_ratio, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

##############################################################################

# plotting the distribution of zeta and wn
z_list_sorted = sorted(z_list)
wn_list_sorted = sorted(wn_list)

# plotting distribution of zeta
sns.distplot(z_list_sorted, hist=True, rug=False, label = 'zeta')
plt.legend(loc = 'best')
plt.xlabel('zeta')
plt.ylabel('probability density')
plt.show()

#p plotting distribution of wn
sns.distplot(wn_list_sorted, hist=True, rug=False, label = 'wn')
plt.legend(loc = 'best')
plt.xlabel('wn')
plt.ylabel('probability density')
plt.show()

##############################################################################

# solve ODEs
for a,b,d in zip(m_list,k_list,c_list):
    y = odeint(model, y0, t, args= (a, b, d))
    y = np.transpose(y, axes=None)
    displacement.append(y[0])

##############################################################################
# concatenating targets together
targets = np.column_stack((wn_list, z_list))
print(targets)

# save data to files
#np.savetxt('ODE_sols.txt', displacement, delimiter= ',')
#np.savetxt('targets.txt', targets, delimiter= ',' )