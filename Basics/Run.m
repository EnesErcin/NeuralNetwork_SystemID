clc; clear; close all;


input_ports     = 5;
test_size       = 2500;
type_of_data    ="sin";
sample_freq     = 100;

[fulldata,preped] =data_prep(type_of_data,input_ports,sample_freq,test_size);

expected_values = preped(1,2:end);
My_model = Neuron_for_sinn(input_ports,test_size);

for i = 1:test_size-1
    My_model.feedforward(preped(:,i),expected_values(:,i),1);
    My_model.backprop();
    disp(My_model.count);
end

validate_size = 250;
results = zeros([validate_size,1]);
compare = zeros([validate_size,1]);

for i = 1:validate_size
    My_model.feedforward(preped(:,i),expected_values(:,i),0);
    results(i)= sum(My_model.y_cache);
end
for i = 1:validate_size
    compare(i) = expected_values(:,i);
end
what = results-compare;


subplot(3,1,1);
plot(compare)

subplot(3,1,2); 
plot(results)

subplot(3,1,3); 
plot(results)
hold on
plot(compare)



