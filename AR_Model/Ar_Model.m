clc; clear; close all;

[x,x_n,t] = data_prep("sin",50,100);

figure;
plot(t(1:100),x(1:100),'b-',t(1:100),x(1:100)-x_n(1:100),'r--');
legend('Signal', 'Noise');
title("Noisy and Original Signal");

deg = 8;
sys  = ar(x_n,deg);
temp = zeros(deg+1,length(x_n));

for i = 1:length(x_n)-deg
        temp(1,i) = 1;
    for j = 2:deg+1
        temp(j,i) = x_n(i+j-1);
    end
end

res = zeros(1,length(x_n));
for i = 1:length(x_n)-deg
    res(i) = sum(temp(:,i).*sys.a');
end

figure;
plot(t(:,(1:end-4)), x_n(:,(1:end-4)), 'b-', t(:,(1:end-4)), 1-res(:,(1:end-4)), 'r-');
legend('Signal with noise', 'AR model output');
xlabel('Time (s)');
ylabel('Amplitude');
title("AR model of order" + deg);

data = iddata((x_n)',[]);
fd = iddata((x_n)',[]);
dtp =100;
yf = forecast(sys,data(1:100),dtp);
figure;
plot(data(1:100),'b',yf,'r'), legend('measured','forecasted');
title("AR model forecasted "+dtp  +" data point ahead");

