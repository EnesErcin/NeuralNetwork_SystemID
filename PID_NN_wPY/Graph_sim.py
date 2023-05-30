import numpy as np
import random
import matplotlib.pyplot as plt

def plot_ss_signals(time_array,store_in_u,stor_out,store_ref,stor_err):
    fig_2, (ax0, ax1)= plt.subplots(2, 1, figsize=(8, 10))

    # Plot data on each subplot
    ax0.plot(time_array, np.array(store_ref)[:,0],label ="Reference Signal",color='r')
    ax0.plot(time_array, np.array(stor_out)[:,0],label  ="Plant Output",color='g')
    ax0.plot(time_array, np.array(store_in_u)[:,0],label ="Control Input")
    ax0.set_title('Reference vs Output')
    ax0.legend()


    ax1.plot(time_array, np.array(stor_err)[:,0],label ="Error[t]",color='r')
    ax1.axhline(y = (np.mean(np.abs(np.array(stor_err)[:,0]))),label ="Abs average error",color='g')
    ax1.set_title('Error')
    ax1.legend()

    plt.show()