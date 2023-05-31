import numpy as np

curr_err = np.zeros((3,1))
curr_ref = np.zeros((3,1))
curr_plant = np.zeros((3,1))
print(np.shape(curr_err))
curr_in = np.concatenate((curr_err,curr_ref,curr_plant))
print(np.shape(curr_in))

curr = [curr_err,curr_in,curr_ref,curr_plant]
for i in curr:
    print(np.shape(i))

curr_in_nn =  np.concatenate(curr_ref,curr_in,curr_plant)