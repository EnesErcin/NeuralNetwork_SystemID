clc; clear; close all;

[x,x_n,t] = data_prep("sin",50,100);

figure;
plot(t(1:100),x(1:100),'b-',t(1:100),x(1:100)-x_n(1:100),'r--');
legend('Signal', 'Noise');
title("Noisy and Original Signal");
noise = x_n -x;

deg = 8;
sys  = ar(x_n,deg-1);
temp = zeros(deg,length(x_n));

for i = 1:length(x_n)-deg
    for j = 1:deg
        temp(j,i) = x_n(i+j);
    end
end

res = zeros(1,length(x_n));

test = (temp(:,4).*sys.a');
for i = 1:length(x_n)-deg
    res(i) = sum(temp(:,i).*sys.a');
end

figure;
subplot(2,1,1);
plot(t(:,(1:end-4)), x(:,(1:end-4)), 'b-', t(:,(1:end-4)), x_n(:,(1:end-4))-res(:,(1:end-4)),'r-', t(:,(1:end-4)),x_n(:,(1:end-4)),"g-.");
legend('Signal without noise', 'AR model output pred', "Signal With Noise");
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t(:,(1:end-4)), noise(:,(1:end-4)), 'b-', t(:,(1:end-4)), res(:,(1:end-4)), 'r-');
legend('Signal noise', 'AR model estimation');
xlabel('Time (s)');
ylabel('Amplitude');


error = abs(x(:,(1:end-4))-x_n(:,(1:end-4)));
error = mean(error);
estimated_error = x(:,(1:end-4))-res(:,(1:end-4));
estimated_error = mean(estimated_error);
fprintf("Reduced average error by Ar model : %d, Initial Error: %d \n",estimated_error,error);
fprintf("System accuracy has improved by %.2f%% \n" , 100*(error-estimated_error)/error);


