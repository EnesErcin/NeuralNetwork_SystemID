from neuron import neuron_layer,neuron_layers
import numpy as np
import matplotlib.pyplot as plt

### Neuraon Network Parameters
hidden_layers=[5,2,4,10,3]
my_layers = neuron_layers(hidden_layers)

print("Number of hidden layers \t {}".format(len(my_layers.layers)))

for i in range (0,len(my_layers.layers)):
    print("Neuron num | {} | \t Input Size | {} | \t Output Size  | {} | \t ".format(i, np.shape(my_layers.layers[i].w)[1], np.shape(my_layers.layers[i].w)[0]) )

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

for i in range (strt_time,endtime+1):
    x_series.append(np.random.randn(hidden_layers[0],1))
    exp_series.append(np.random.randn(hidden_layers[-1],1))

for i in range (strt_time,endtime+1):
    result = my_layers.calc_output(x_series[i],exp_series[i])
    output.append(result[0])
    error.append(result[1])

print("---------------------------------------------------------------------------------------------------------")
print("Time array shape \t {} \nError and output shape \t {} \t {} \nX shape \t {}".format(np.shape(time_array),np.shape(output),np.shape(error),np.shape(x_series)))

output = np.array(output)
error = np.array(error)
x_series = np.array(x_series)
exp_series =np.array(exp_series)


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

plt.show()