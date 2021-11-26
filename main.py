import numpy as np
import matplotlib.pyplot as plt
from Pulse_UTILITY import pulse_utility
t = np.linspace(-50e-9,50e-9,2000)
sigma = 1e-9;
tp = 1e-9;
inst = pulse_utility()
pulse= inst.gaussian(t,0,sigma)*np.exp(1j*t**2/tp)
test = inst.fourier_transform(t,pulse)
freq = test[0]
test2 = inst.inverse_fourier_transform(test[0],test[1])

power_spectrum = inst.get_power_spectrum(t,pulse)
tau , autocorrelation = inst.get_amplitude_autocorrelation(t,pulse)
tau2, G2              = inst.get_intensity_autocorrelation(t,pulse)


fig, ax = plt.subplots(3)
ax[0].plot(t,np.abs(pulse)**2)
ax[0].axis(xmin = -5*sigma, xmax = 5*sigma)
ax[0].set_xlabel('Time (s)')
ax[1].plot(freq, power_spectrum)
ax[1].axis(xmin = -8/(5*sigma),xmax = 8/(5*sigma))
ax[1].set_xlabel('Frequency (Hz)')
#ax[1].plot(test2[0], test2[1]**2)
ax[2].plot(tau, np.abs(autocorrelation))
ax[2].axis(xmin = -5*sigma, xmax = 5*sigma)
ax[2].set_xlabel('Time Delay (s)')
#ax[3].plot(tau, G2)
#ax[3].axis(xmin = -5*sigma, xmax = 5*sigma)
#ax[3].set_xlabel('Time Delay (s)')
plt.show()