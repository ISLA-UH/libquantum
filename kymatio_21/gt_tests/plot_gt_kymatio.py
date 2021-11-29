"""
Compute the scattering transform of a synthetic signal
======================================================

In this example we generate a harmonic signal of a few different frequencies
and analyze it with the 1D scattering transform.

See plot_synthetic_mag
"""


###############################################################################
# Import the necessary packages
# -----------------------------
from kymatio.numpy import Scattering1D
import matplotlib.pyplot as plt
import numpy as np
import kymatio_21.kymatio_atom_utils as atom

###############################################################################
# Let's take a look at what such a signal could look like
# -------------------------------------------------------

Q = 6
sample_rate = 8000.

# Min spec of 2**10, critically sampled
log2T = 10
T = 2**log2T

###############################################################################
# Prepare the scattering transform
# ------------------------------

x, gt_J, gt_frequency_hz = \
    atom.gt_blast_kymatio(time_points=T,
                          nth_octave=Q,
                          sig_octaves_up=0,
                          noise_bit_loss=6,
                          sample_rate_hz=sample_rate)

J = gt_J
# Print averaging frequency
averaging_period = 2**J/sample_rate
averaging_frequency = 1/averaging_period
print("J=", J)
print("Averaging frequency:", averaging_frequency)
print("Peak frequency:", gt_frequency_hz)

# # Inspect source code: in filter_bank.py
# sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = \
#     calibrate_scattering_filters(J, Q, r_psi=np.sqrt(0.5), sigma0=0.1, alpha=5.)
# frequency_xi1 = np.array(xi1)*sample_rate
# frequency_xi2 = np.array(xi2)*sample_rate

# Compute scattering operation parameters
# Kymatio scattering tensor
scattering = Scattering1D(J, T, Q)

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

# Use meta method to extract scaled frequencies
xi = meta['xi']
# Convert normalized frequency xi = f/fs to frequency
freqs = xi*sample_rate

# Scattering transform
Sx = scattering(x)
print("Scattering tensor shape:", Sx.shape)

# This is what is used to
# Split the solutions into zeroth, first, and second order
Sx_0 = Sx[order0][0]
Sx_1 = Sx[order1]
Sx_2 = Sx[order2]
frequency_0 = freqs[order0, 0]
frequency_1 = np.squeeze(freqs[order1, 0])
frequency_2 = freqs[order2]
time_len = len(Sx_0)
decimation_factor = len(x)/time_len

time_x = np.arange(len(x))/sample_rate
time_Sx = np.arange(len(Sx_0))/sample_rate*decimation_factor

# # Print the zeroth, first, and second order solutions
# print(frequency_0)
# print(frequency_1)
# print(frequency_2)
# print(time_len, decimation_factor)

# x = frequency_2[:, 0]
# y = frequency_2[:, 1]
# print(x)
# print(y)

# Time series
plt.figure(figsize=(8, 2))
plt.plot(time_x, x)
plt.title("Original signal")

# Scattering panels
plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(Sx_0)
plt.title('Zeroth-order scattering')
plt.subplot(3, 1, 2)
plt.imshow(Sx_1, aspect='auto')
plt.title('First-order scattering')
plt.subplot(3, 1, 3)
plt.imshow(Sx_2, aspect='auto')
plt.title('Second-order scattering')

# Compare to first order to spectrogram
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.pcolormesh(time_Sx, frequency_1, Sx_1, shading='auto')
plt.ylim(np.min(frequency_1), np.max(frequency_1))
plt.title("First order")
plt.subplot(1, 2, 2)
plt.specgram(x, Fs=sample_rate)
plt.ylim(np.min(frequency_1), np.max(frequency_1))
plt.title('Spectrogram')

# Use scatter plot for second order?


plt.show()
