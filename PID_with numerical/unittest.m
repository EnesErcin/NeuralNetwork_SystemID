clear;clc;

sr = 1;
endtime = 50;

time_array = 1:sr:endtime;
my_intg = Intg(0,sr);
my_diff = Diff(0,sr);
my_PID = PID_CNTRL(0,sr);

% Initialize output array
% y = increasingStepFunction(time_array,sr,endtime);

res_int = zeros(size(time_array));
res_dif = zeros(size(time_array));

squared_values = time_array.^2;
y = squared_values;

for t = 1:sr:endtime
   
    [my_intg,res_int(t)] = my_intg.intg(y(t));
    [my_diff,res_dif(t)] = my_diff.diff(y(t));
end





