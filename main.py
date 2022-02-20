import numpy as np
import matplotlib.pyplot as plt
import random
from Pulse_UTILITY import pulse_utility

#Define Pulse Parameters
length = 400000
t = np.linspace(-500,500,length)
dt = t[1] - t[0]
tp = 1
tc = 200000
w0 = 2*np.pi

phi_rand = np.zeros(np.shape(t))
random_samples = 10*np.random.normal(0, 1,length-1)
plt.show()
plt.hist(random_samples)
for i in range(length -1):
    #phi_rand[i+1] = phi_rand[i] + 5*np.random.uniform(-np.pi,np.pi)*dt
    phi_rand[i + 1] = phi_rand[i] + random_samples[i]* dt



inst = pulse_utility()

#Find Pulse in time and frequency domain
pulse_stable = inst.gaussian(t,0,tp/np.sqrt(2))*np.exp(1j*(w0*t))
pulse_rand = inst.gaussian(t,0,tp/np.sqrt(2))*np.exp(1j*(w0*t + phi_rand))


freq, phi_freq = inst.fourier_transform(t,np.exp(1j*phi_rand))
freq, pulse_fdom = inst.fourier_transform(t,pulse_stable)
freq, pulse_rand_fdom = inst.fourier_transform(t,pulse_rand)


#Caluculate Intensity FWHM and TB Product
#FWHM_time = inst.get_FWHM(t,pulse_stable)
#FWHM_freq = inst.get_FWHM(freq,pulse_fdom)
#TB_product = FWHM_freq*FWHM_time


fig, ax = plt.subplots(3)
ax[0].plot(t,phi_rand)
ax[0].set_title('Random Phase (Time Domain)')
ax[1].plot(t,np.abs(phi_freq)**2)
ax[1].set_xlim(-5/(tp),5/tp)
ax[1].set_title('Phase Power Spectrum (Freq Domain)')
ax[2].plot(freq,np.abs(pulse_fdom)**2, color = 'b', label = 'Zero Phase')
ax[2].plot(freq,np.abs(pulse_rand_fdom)**2, color = 'orange', label = 'Random Phase')
ax[2].legend()
ax[2].set_xlim(-5/(tp),5/tp)
ax[2].set_title('Gaussian Pulse Power Spectrum')
plt.show()
#Plotting
#fig, ax = plt.subplots(2)
#fig.suptitle('$\Delta \\nu \Delta t$: {}'.format(FWHM_time*FWHM_freq))
#ax[0].plot(t, np.abs(pulse)**2, label = '$\Delta t =$ {}'.format(FWHM_time))
#ax[0].plot(t2, np.abs(pulse2)**2,label = '$\Delta t =$ {}'.format(FWHM_time))
#ax[0].axvspan(-FWHM_time/2, FWHM_time/2, facecolor='g', alpha=0.5)
#ax[0].set_xlim([-5*tp,5*tp])
#ax[0].legend()


#ax[1].plot(freq, np.abs(pulse_fdom)**2, label = '$\Delta \\nu$: {}'.format(FWHM_freq))
#ax[1].set_xlim(-5/(tp),5/tp)
#ax[1].axvspan(-FWHM_freq/2, FWHM_freq/2, facecolor='r', alpha=0.5)
#ax[1].legend()
#plt.show()
