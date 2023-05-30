import numpy as np
import matplotlib.pyplot as plt
import random

## Import Neuron Network 
from Neuron.neuron import neuron_layer,neuron_layers
from PID.pid_cntrl import PID_CNTRL
from System_Plant.system_input import _input_gen
from System_Plant.plant import Plant

#####################################
#### Neuron network parameter #######
#####################################

#############################
### Inputs || Outputs #######
#############################

## Error        [t_0,t_1,t_2]
## Reference    [t_0,t_1,t_2]
## Plant_Output [t_0,t_1,t_2]
# -> Number of inputs = 3*3 = 9
## Output [Kp,Kd,Ki]
# -> Number of outputs = 3

curr_err = np.zeros((3,1))
curr_ref = np.zeros((3,1))
curr_plant = np.zeros((3,1))
curr_in = np.concatenate((curr_err,curr_ref,curr_plant),axis=0)

nn_in_cnt = np.shape(curr_in)[0]

curr_nn_out = np.zeros((3,1))
nn_out_cnt = np.shape(curr_nn_out)[0]

# Number of nodes in each layer
hidden_layers=[nn_in_cnt,4,5,3,nn_out_cnt]
# Learning rate
lr = 0.03

## Initiate neuron network layer
Deep_NN = neuron_layers(hidden_layers,lr)


################################################


########################################
#### Initate PID controller & Plant ####
########################################

# Initial pid coeffs
Kp = 0.8
Kd = 0.4
Ki = 0.2
## Initiate PID controller
Pid = PID_CNTRL(Kp=Kp,Kd=Kd,Ki=Ki)

System_Plant = Plant(prev_val=0)




""""
learning_rate = 5*10**-5

print("Number of hidden layers \t {}".format(len(my_layers.layers)))

for i in range (0,len(my_layers.layers)):
    print("Neuron num | {} | \t Input Size | {} | \t Output Size  | {} | \t ".format(i, np.shape(my_layers.layers[i].w)[1], np.shape(my_layers.layers[i].w)[0]) )
    print("\nNeuron num | {} | \t Shape | {} | \t ".format(i, np.shape(my_layers.layers[i].w)))

### Simulations parameters

#### TIME
# Sample time
st = 1
# Start- End time 
strt_time = 0
endtime = 100
# Time array 
time_array = np.arange(strt_time, endtime+st, st)

### Data
x_series   = [] ## Input
smpl_cnt   =  int( ( strt_time - endtime ) / st ) 
exp_series = [] ## Expected

### Store the output
output = []
error  = []


my_data_set_1 = []
my_data_set_2 = []
my_data_set_3 = []

my_expected_val_1 = []
my_expected_val_2 = []
my_expected_val_3 = []

def _fnc(x,num):
    if num == 1:
        return 10*i+4
    elif num == 2:
        if i %6 :
            a= np.sin(i) + i**0.1
        elif i %3:
            a= np.cos(i)*10
        else:
            a = i
        return a
    else:
        return 10-20*i

for i in range (strt_time,endtime+1):
    val_1 = (_fnc(i,1))
    val_2 = (_fnc(i,2))
    val_3 = (_fnc(i,3))

    arr = np.array([val_1,val_2,val_3]).T
    print(np.shape(arr)[0])
    arr = np.reshape(arr,(np.shape(arr)[0],1))

    arr_2 = np.array([i,i,i]).T
    arr_2 = np.reshape(arr_2,(np.shape(arr_2)[0],1))

    my_expected_val_1.append(arr)
    my_data_set_1.append(arr_2)

for i in range (strt_time,endtime+1):
    ## Shape x_series -> (hidden_layers[0],1)
    ## EXP x_series ->   (hidden_layers[-1],1)
    x_series.append(np.random.randn(hidden_layers[0],1))
    exp_series.append(np.random.randn(hidden_layers[1],1))

#print("\n\n\n")
#print(type(x_series))
#print(type(my_data_set_1))
#print(x_series[5])
#print(my_data_set_1[-1])
#print(np.shape(x_series[5]))
#print(np.shape(my_data_set_1[-1]))
#assert False

w = []

for i in range (strt_time,endtime+1):
    ## Feed forward
    result = my_layers.calc_output(my_data_set_1[i],my_expected_val_1[i])

    ## Backprop
    my_layers.back_prop()

    ## Save the results
    output.append(result[1])
    error.append(result[0])
    w.append(my_layers.layers[2].w)

print("---------------------------------------------------------------------------------------------------------")
print("Time array shape \t {} \nError and output shape \t {} \t {} \nX shape \t {}".format(np.shape(time_array),np.shape(output),np.shape(error),np.shape(x_series)))

output = np.array(output)
error = np.array(error)
x_series = np.array(my_data_set_1)
exp_series =np.array(my_expected_val_1)
w =np.array(w)

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# Plot data on each subplot
axs[0].plot(time_array, output[:,1])
axs[0].set_title('Outputs')

axs[1].plot(time_array, error[:,1])
axs[1].set_title('Errors')

axs[2].plot(time_array, x_series[:,1])
axs[2].set_title('Inputs ')

axs[3].plot(time_array, exp_series[:,1])
axs[3].set_title('Expected values ')


# Create subplots
fig_2, axs_2 = plt.subplots(4, 1, figsize=(8, 10))

axs_2[0].plot(time_array, output[:,2])
axs_2[0].set_title('Outputs')

axs_2[1].plot(time_array, error[:,2])
axs_2[1].set_title('Errors')

axs_2[2].plot(time_array, x_series[:,2])
axs_2[2].set_title('Inputs ')

axs_2[3].plot(time_array, exp_series[:,2])
axs_2[3].set_title('Expected values ')



# Create subplots
fig__3, axs_3 = plt.subplots(5, 1, figsize=(8, 10))

axs_3[0].plot(time_array, output[:,0])
axs_3[0].set_title('Outputs')
axs_3[1].plot(time_array, error[:,0])
axs_3[1].set_title('Errors')
axs_3[2].plot(time_array, x_series[:,0])
axs_3[2].set_title('Inputs ')
axs_3[3].plot(time_array, exp_series[:,0])
axs_3[3].set_title('Expected values ')

axs_3[4].plot(time_array, w[:,0])
axs_3[4].set_title('Weigths ')

plt.show()

"""