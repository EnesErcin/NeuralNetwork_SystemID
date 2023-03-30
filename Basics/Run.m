clc; clear; close all;


input_ports     = 5;
test_size       = 250;
type_of_data    ="sin";
sample_freq     = 100;

[fulldata,preped] =data_prep(type_of_data,input_ports,sample_freq,test_size);

expected_values = preped(1,2:end);


My_model = Neuron_for_sinn(input_ports);

for i = 1:test_size-1
    My_model.feedforward(preped(:,i),expected_values(:,i));
    My_model.backprop();
end

