import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg

# Define the sampling rate
fs = 300  # Hz
# Generate a time vector
t = np.arange(0, 1, 1/fs)  # 1 second of data
# Generate a sine wave
f0 = 10  # Hz
x = np.sin(2*np.pi*f0*t)

# Add noise
noise_std = 0.1  # standard deviation of the noise
noise = np.random.normal(0, noise_std, len(x))
x_noisy = x + noise

dur = 300
train = x_noisy[:dur-50]
test  = x_noisy[dur-50:]

model = AutoReg(train,lags=5).fit()

pred = model.predict(start=len(train), end = 299)

pyplot.plot(pred)
pyplot.plot(test,color = "red")
pyplot.show()
