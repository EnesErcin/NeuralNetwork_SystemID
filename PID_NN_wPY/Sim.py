from System_Params import Pid,Deep_NN,System_Plant
from System_Params import curr_err,curr_ref,curr_in,curr_plant,curr_nn_out
from System_Plant.system_input import _input_gen
from Graph_sim import plot_ss_signals
import numpy as np


#############################
### Simulation paramaters ###
#############################

# Linear system ?
is_linear = System_Plant.is_linear

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
update_k = False
temp = 0

######################################
###### Store simulation results  #####
######################################
stor_err    = np.zeros((np.size(sim_t),1))
stor_out    = np.zeros((np.size(sim_t),1))
store_ref   = np.zeros((np.size(sim_t),1))
store_in_u  = np.zeros((np.size(sim_t),1))
###################################
############## RUN SIM ############
###################################

for t in range (sim_s,sim_end):
    # Run the simulation for one itteration

    u_signal = Pid._proc(temp)

    if update_k:
        Pid._update_k()
    else:
        pass

    out_t_1_signal= System_Plant._proc(u_signal)

    error_t_1_signal = ref[t+1] - out_t_1_signal

    temp = error_t_1_signal


    #### Store simulation results at every itteration
    #################################################
    stor_err[t]  = error_t_1_signal
    stor_out[t]  = out_t_1_signal
    store_ref[t] = ref[t+1]
    store_in_u[t]=u_signal

plot_ss_signals(is_linear,sim_t,store_in_u,stor_out,store_ref,stor_err)