import numpy as np
import matplotlib.pyplot as plt
from Pulse_UTILITY import pulse_utility
t = np.linspace(-500,500,400000)
tp = 1
tc = 200000
w0 = 2*np.pi
inst = pulse_utility()
#pulse = 1/(np.cosh(t/tp))
pulse = inst.gaussian(t,0,tp/np.sqrt(2))
#pulse2 = inst.gaussian(t,0,10*tp/np.sqrt(2))
pulse3 = pulse
#pulse3 = np.sqrt(0.1)*pulse + np.sqrt(0.9)*pulse2

freq, pulse_fdom = inst.fourier_transform(t,pulse3)

FWHM_time = inst.get_FWHM(t,pulse3)
FWHM_freq = inst.get_FWHM(freq,pulse_fdom)
TB_product = FWHM_freq*FWHM_time


fig, ax = plt.subplots(2)
fig.suptitle('$\Delta \\nu \Delta t$: {}'.format(FWHM_time*FWHM_freq))
ax[0].plot(t, np.abs(pulse3)**2, label = '$\Delta t =$ {}'.format(FWHM_time))
ax[0].axvspan(-FWHM_time/2, FWHM_time/2, facecolor='g', alpha=0.5)
ax[0].set_xlim([-5*tp,5*tp])
ax[0].legend()


ax[1].plot(freq, np.abs(pulse_fdom)**2, label = '$\Delta \\nu$: {}'.format(FWHM_freq))
ax[1].set_xlim(-5/(tp),5/tp)
ax[1].axvspan(-FWHM_freq/2, FWHM_freq/2, facecolor='r', alpha=0.5)
ax[1].legend()
plt.show()
