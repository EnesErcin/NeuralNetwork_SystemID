from System_Params import Pid,Deep_NN,System_Plant
from System_Params import curr_err,curr_ref,curr_in,curr_plant,curr_nn_out,curr_in_nn
from System_Params import Nn_pid_intrf
from System_Plant.system_input import _input_gen
from Graph_sim import plot_ss_signals
from Graph_sim import _fifo
import numpy as np


#############################
### Simulation paramaters ###
#############################

# Linear system ?
is_linear = System_Plant.is_linear

# With Deep Neural Network
w_NN = True

# Define a simulation time array
sim_s = 0
sim_end = 300
sr = 1
assert(sr==1) # Do not change step size not yet
assert(sim_end>sim_s) 

sim_t = np.arange(sim_s,sim_end,sr)

# Generate reference signal
ref = _input_gen(is_linear,sim_t,sr,sim_end,sim_s)

# Make update_k True to enable Deep NN optimizations
update_k = w_NN
temp = 0

######################################
###### Store simulation results  #####
######################################
stor_err    = np.zeros((np.size(sim_t),1))
stor_out    = np.zeros((np.size(sim_t),1))
store_ref   = np.zeros((np.size(sim_t),1))
store_in_u  = np.zeros((np.size(sim_t),1))
store_k_coef= np.zeros((np.size(sim_t),3))
###################################
############## RUN SIM ############
###################################

error_t_1_signal = 0
out_t_1_signal = 0


for t in range (sim_s,sim_end):
    # Run the simulation for one itteration

    u_signal = Pid._proc(temp)
    
    k_coefs = Deep_NN.forward(curr_in_nn)
    k_coefs = np.clip(k_coefs,-3,3)

    if update_k:
        Pid._update_k(k_coefs[0],k_coefs[1],k_coefs[2])        
    else:
        pass

    out_t_1_signal = System_Plant._proc(u_signal)

    error_t_1_signal = ref[t+1] - out_t_1_signal
    
    if update_k:
        dJ_da = Nn_pid_intrf._proc(new_e=error_t_1_signal)
        Deep_NN.back_prop(dJ_da)
    
    temp = error_t_1_signal
    
    ### Store delayed values as DNN input signal
    
    curr_err = _fifo(curr_err,error_t_1_signal)
    curr_in  =  _fifo(curr_in,u_signal)
    curr_plant = _fifo(curr_plant,out_t_1_signal)
    curr_ref = _fifo(curr_ref,ref[t+1])

    curr = [curr_err,curr_in,curr_ref,curr_plant]
    curr_in_nn =  np.concatenate((curr_ref,curr_in,curr_plant))
    curr_in_nn = np.concatenate((curr_in_nn,curr_err))
    
    curr_in_nn = np.reshape(curr_in_nn,(np.shape(curr_in_nn)[0],1))
    

    #### Store simulation results at every itteration
    #################################################
    stor_err[t]  = error_t_1_signal
    stor_out[t]  = out_t_1_signal
    store_ref[t] = ref[t+1]
    store_in_u[t]=u_signal

    store_k_coef[t,:]= np.reshape(np.array(k_coefs),(1,3))

plot_ss_signals(is_linear,w_NN,sim_t,store_in_u,stor_out,store_ref,stor_err,store_k_coef)
