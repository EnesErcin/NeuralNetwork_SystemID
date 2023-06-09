import numpy as np
import matplotlib.pyplot as plt
import random

## Import Neuron Network 
from Neuron.neuron import neuron_layer,neuron_layers
from PID.pid_cntrl import PID_CNTRL
from System_Plant.system_input import _input_gen
from System_Plant.plant import Plant
from PID.NN_PID_intrface import NN_PID_interface

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
curr_in = np.zeros((3,1))
curr_in_nn = np.concatenate((curr_err,curr_ref,curr_plant,curr_in),axis=0)

nn_in_cnt = np.shape(curr_in_nn)[0]

curr_nn_out = np.zeros((3,1))
nn_out_cnt = np.shape(curr_nn_out)[0]

# Number of nodes in each layer
hidden_layers=[nn_in_cnt,4,5,3,nn_out_cnt]
# Learning rate
lr = 0.003

## Initiate neuron network layer
update_method = "ref" 
assert update_method == "ref" # For this sytem it should only work with ref
Deep_NN = neuron_layers(hidden_layers,lr,update_method)


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

# To pick a non linear system make >>
# is_linear = True for linear system
is_linear = True
System_Plant = Plant(prev_val_y=0,is_linear=is_linear)


## Nerual Network -- PID interface module gen

Nn_pid_intrf  = NN_PID_interface(is_linear=is_linear)



