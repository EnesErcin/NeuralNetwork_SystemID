clear;clc;

% Simulation parameters defined
sr = 1; % Sample Rate
endtime = 180; % Simulation End time
time_array = 1:sr:endtime;

% Pid and Plant Instentation
my_PID = PID_CNTRL(0,sr);
my_Plant = RandomPlant(0);

% Reference Signal Generation
ref = rndm_input_gen(time_array,sr,endtime,40);

% Arrays to store output values of the system
sys_u = zeros(size(time_array));
sys_y = zeros(size(time_array));
sys_e = zeros(size(time_array));

new_e = 0;
for t = 1:sr:endtime
        %System process 
        [my_PID,u] = my_PID.proc(new_e);
        [my_Plant,res] = my_Plant.proc(u);

        new_e = ref(t) - res;
        
        %Store the values
        sys_u(t) = u;
        sys_y(t) = res; 
        sys_e(t) = new_e; 
end






