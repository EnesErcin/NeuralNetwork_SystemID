import numpy as np
import random
import matplotlib.pyplot as plt

def plot_ss_signals(is_linear,wNN,time_array,store_in_u,stor_out,store_ref,stor_err,k_coefs):
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

    # Set the desired directory path

    if wNN:
        directory_path = 'Sim_pics/wNN/'
    else:
        directory_path = 'Sim_pics/woNN/'
    
    if is_linear :
        image_str = 'Sim_{}_output.png'.format("linear")
    else:
        image_str = 'Sim_{}_output.png'.format("non_linear")



    # Save the plot in the specified directory
    plt.savefig(directory_path + image_str)

    plt.show()
    
    fig_3, (ax0, ax1)= plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot data on each subplot
    ax0.plot(time_array, np.array(k_coefs)[:,0],label ="Kp",color='r')
    ax0.plot(time_array, np.array(k_coefs)[:,1],label  ="Kd",color='g')
    ax0.plot(time_array, np.array(k_coefs)[:,2],label ="Ki")
    ax0.set_title('Reference vs Output')
    ax0.legend()


    ax1.plot(time_array, np.array(stor_err)[:,0],label ="Error[t]",color='r')
    ax1.axhline(y = (np.mean(np.abs(np.array(stor_err)[:,0]))),label ="Abs average error",color='g')
    ax1.set_title('Error')
    ax1.legend()

    # Set the desired directory path

    if wNN:
        directory_path = 'Sim_pics/wNN/'
    else:
        directory_path = 'Sim_pics/woNN/'
    
    if is_linear :
        image_str = 'K_coefs_{}_output.png'.format("linear")
    else:
        image_str = 'K_coefs_{}_output.png'.format("non_linear")


    # Save the plot in the specified directory
    plt.savefig(directory_path + image_str)

    plt.show()


def _fifo(arr,new_val):
    arr = np.delete(arr,np.size(arr)-1)
    arr = np.append(arr,new_val)
    #arr = np.reshape(arr,(np.shape(arr)[0],1))
    return arr