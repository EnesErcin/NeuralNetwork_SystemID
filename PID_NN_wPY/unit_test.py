import numpy as np

curr_err = np.zeros((3,1))
curr_ref = np.zeros((3,1))
curr_plant = np.zeros((3,1))
print(np.shape(curr_err))
curr_in = np.concatenate(curr_err,curr_ref)