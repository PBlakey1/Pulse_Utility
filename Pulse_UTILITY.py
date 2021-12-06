import numpy as np
from scipy.fft import fft, fftfreq, fftshift,ifft,ifftshift,ifftn


class pulse_utility:
    def gaussian(self,x,mu,sigma):
        result = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2/(2*sigma**2))
        return result

    def fourier_transform(self,t,a):
        N = np.size(t)
        T = max(t)/N
        f_domain = fftshift(fft(a))
        freq = fftfreq(N,T)
        freq = fftshift(freq)
        return freq, f_domain

    def inverse_fourier_transform(self,freq, a_f):
        N = np.size(freq)
        F_samp = max(freq) / N
        a_t = ifftshift(ifft(a_f))
        t = fftfreq(N, F_samp)
        t = ifftshift(t)
        return t, a_t

    def get_FWHM(self,x,a):
        a = np.abs(a)**2
        max_index = np.argmax(a)
        idx = (np.abs(a - a[max_index]/2)).argmin()
        FWHM = x[max_index]-x[idx]
        return 2*FWHM
    def get_power_spectrum(self,t,a):
        power_spectrum = np.abs(self.fourier_transform(t,a)[1])**2
        return power_spectrum

    def get_amplitude_autocorrelation(self,t,a):
        freq , a_f = self.fourier_transform(t,a)
        power_spect = self.get_power_spectrum(t,a)
        tau, autocorrelation = self.inverse_fourier_transform(freq,power_spect)
        #autocorrelation = np.convolve(a,np.conj(a),'same')
        return tau, autocorrelation

    def get_intensity_autocorrelation(self,t,a):
        dt = np.mean(np.diff(t))
        freq, I_f = self.fourier_transform(t, np.abs(a)**2)
        tau, conv = self.inverse_fourier_transform(freq,I_f**2*np.exp(1j*freq*t))
        norm = sum(np.abs(a)**2)*dt
        G2 = np.divide(np.abs(conv),norm)
        return tau, G2