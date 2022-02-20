import numpy as np
import matplotlib.pyplot as plt
from Pulse_UTILITY import pulse_utility

t = np.linspace(-5000,5000,500000)
tp = 1
tc = 0.5
w0 = 10;

inst = pulse_utility()

pulse_td = np.real(inst.gaussian(t,0,tp/np.sqrt(2))*np.exp(1j*w0*t))

tau, Gamma = inst.get_amplitude_autocorrelation(t,pulse_td)

plt.plot(tau, np.abs(Gamma)**2)
plt.xlim([-5*tp,5*tp])
plt.show()