function [x,x_n,t] = data_prep(type,f0,fs)
    if type == "sin"
        % Define the sampling rate
        fs = 1000;  % Hz

        % Generate a time vector
        t = 0:1/fs:1-1/fs;  % 1 second of data

        % Generate a sine wave
        x = sin(2*pi*f0*t);
        
        noise_std = 0.1;  % standard deviation of the noise
        noise = noise_std*randn(1,length(t));
        x_n=  noise + x;
    end
end
