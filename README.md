# NeuralNetwork_SystemID
Applications of neural networks on system identification special cases for control end electrical engineering.


## Introduction

PID Tunining practice with neural network [resarch paper](https://ieeexplore.ieee.org/document/714292).

## Issues

It is important to mention this application is not without flaws. There are some mistakes in this code. There are some issues regarding mathematical derivation as well. In the end with slight workaround solutions, the outcome is improved in the linear system by almost %20. While a non-linear system becomes unstable by disproportional tuning of the PID controller. More detailed mathematical derivations have to be made to improve the systems.

The architecture of the system is also debatable. PID tuning with the neural network may seem a reasonable idea but input adjustments of the pid tuner might be a better structure change to the system. This means Deep NN can be cascaded attached to the PID rather than adjusting coefficients. This is not yet tried simply because this was the homework requirement, another reason is some advantageous of this application. One such is the user can later disable the deep neural network structure from the system to improve power consumption and computational workload.

In conclusion, do not hesitate to clarify the issues represented in this code. More pleasantly do not hesitate to make contributes to the code.
## Required packages

Installation of numpy (Matrix calculation libary):

    $ pip install numpy

Installation of Matplotlib (Visiual Graphing libary):

    $ pip install matplotlib

Installation of Logging (Easy logging libary):

    $ pip install logging

To run the simulation run the python file named (`\Sim.py`)
To try out different parameters check (`\System_Params`)
To check out derivation of calculations used in this code check (`\PID_Tuning_wDNN.pdf`) or referenced resarch paper.

#### Signals
(/Sim.py)<br>
Simulation signals can be tracked in the file (`Sim.py`):
* `ref[t]`: Reference signal  @ sim_time: t
* `u_signal[t]`: PID Output Signal @ sim_time: t
* `k_coefs[t]`: Pid coefficents @ sim_time: t
* `error_t_1_signal[t]`: Reference - Plant output  @ sim_time: t
* `out_t_1_signal[t]`: Plant output @ sim_time: t

#### Constructor parameters:
(/System_Params.py)<br>
Simulation parameters are modifable to some degree in the file `System_Params.py`:

* `is_linear`: Paper describes two different systems. Non-linear or Linear. Make this parameter True for linear choice and false for otherwise. (type : bool)
* `w_NN`: Enable neural network to update coefficents. w_NN = True to enable NN, false for bypassing the NN. (type: bool)
* `lr` : learning rate of the neural network. (type : float)
* `hidden_layers`: Number of nodes in each layer (type : list)
Restriction being:
It should be an list, 1st index should be 12, last index should be 3.hidden_layers[0] = 12 hidden_layers[-1] = 3
* `Kp`: Inital propotional coefficent of the PID. (type : float)
* `Ki`: Initial integralcoefficent of the PID (type : float)
* `Kd`: Initial differential coefficent of the PID (type : float)

#### Objects:
(/PID)<br>
* `numeric_dif()` :Containts attributes of `prev_val` calculates the difference with **Limit derivation of derivative**. Method of this object is `diff()` which calculates the differential result of the given values regarding the previous value.
* `numeric_intg()` : Containts attributes of `acc` and `prev_val` calculates the integral with **Trapozoidal Method** . Method of this object is `intg()` which calculates the differential result of the given values regarding the previous value.
* `PID_CNTRL()` : Containts `numeric_intg` and  `numeric_dif()` objects as attributes. Two method exist for this object `_proc()` and `update_k()`. First method calculates output of the PID by described methods of numeric methods of differential and integrations. 
(/Neuron)
* `Neuron_layer()`: Single neuron layer. Contains the method of `forward()` and `backprop()`: First method mentioned is foward propagation, and secondone is backpropagation. Needed variables to calculate partial differentials are stored in class variables such as attributes `self.w`, `self.x`, `self.b`.
* `Neuron_Layers()`: Contains `Neuron_layer()` objects. Methods of neuron_layer are `calc_output()` and `backprop()`. Backprop calculates partial derivative of each weigths and biases with reverse direction by calling `Neuron_layer().backprop()` of single layers backproapgation method one by one. Calculate output method is simallry do this in forward direction to return output of the neuron depedning on the output.