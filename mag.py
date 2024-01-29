import numpy as np
import matplotlib.pyplot as plt
fD = 10
Pr = 1

# Parameters
N = 10000  # Number of samples
bins = 50  # Number of bins for histogram

# Generate Rayleigh fading channel coefficients
X = np.random.normal(0, np.sqrt(0.5), N)
Y = np.random.normal(0, np.sqrt(0.5), N)
h = X + 1j*Y

#1. Plot histogram of real and imaginary parts
plt.figure()
plt.subplot(1, 2, 1)
plt.hist(X, bins)
plt.title('Histogram of Real Part (X)')
plt.xlabel('Real Part (X)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(Y, bins)
plt.title('Histogram of Imaginary Part (Y)')
plt.xlabel('Imaginary Part (Y)')
plt.ylabel('Frequency')
plt.show()

# 2. Plot histogram of magnitude of fading coefficient
plt.figure()
plt.hist(np.abs(h), bins)
plt.title('Histogram of Magnitude of Fading Coefficient (|h|)')
plt.xlabel('Magnitude (|h|)')
plt.ylabel('Frequency')
plt.show()

#3. Plot histogram of power of fading coefficient
plt.figure()
plt.hist(np.abs(h)**2, bins)
plt.title('Histogram of Power of Fading Coefficient (|h|^2)')
plt.xlabel('Power (|h|^2)')
plt.ylabel('Frequency')
plt.show()

#4. Find and plot the autocorrelation of real and imaginary parts separately
def hc(fD, Pr):
    freq_values = np.arange(-0.985*fD, 0.985*fD, 0.25)
    Hf_values = []
    for f in freq_values:
        ratio = f / fD
        square_root_term = np.sqrt(1 - ratio**2)
        denominator = np.pi * fD * square_root_term
        Hf_value = np.sqrt((2 * Pr) / denominator)
        Hf_values.append(Hf_value)
        
    return np.fft.ifft(Hf_values)

htime = hc(fD, Pr)
h_conv= np.convolve(h, htime)

def autocorrelation(Xi, max_lag=50):
    N = len(Xi)
    xi_mean = np.mean(Xi)
    xi_sum_squares = np.sum((Xi - xi_mean) ** 2)
    autocorr_values = np.zeros(max_lag)

    for lag in range(max_lag):
        xi_lag = Xi[lag:]
        xi_shifted = Xi[:N-lag]
        autocorr_values[lag] = np.sum((xi_lag - xi_mean) * (xi_shifted - xi_mean)) / xi_sum_squares

    return autocorr_values

acorr_r = autocorrelation(h_conv.real)
acorr_imag = autocorrelation(h_conv.imag)
plt.figure()
plt.plot(acorr_r)
plt.title('Autocorrelation of Real Part')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()


plt.figure()
plt.plot(acorr_imag)
plt.title('Autocorrelation of Imaginary Part')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()


# 5. Find and plot the PSD of real and imaginary parts separately
psd_r=np.fft.fft(acorr_r)
psd_imag=np.fft.fft(acorr_imag)
plt.figure()
plt.plot(psd_r)
plt.title('Power Spectral Density')
plt.xlabel('Index')
plt.ylabel('PSD of Real Part')
plt.show()

plt.figure()
plt.plot(psd_imag)
plt.title('Power Spectral Density')
plt.xlabel('Index')
plt.ylabel('PSD of Imaginary Part')
plt.show()

