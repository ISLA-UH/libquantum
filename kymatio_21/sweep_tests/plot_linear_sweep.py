"""
Compute the scattering transform of a synthetic signal
======================================================

In this example we generate a harmonic signal of a few different frequencies
and analyze it with the 1D scattering transform.
"""
# SEE: https://githubmemory.com/repo/kymatio/kymatio/issues/713

###############################################################################
# Import the necessary packages
# -----------------------------
from kymatio.numpy import Scattering1D
from kymatio.scattering1d.filter_bank import calibrate_scattering_filters
import matplotlib.pyplot as plt
import numpy as np
from libquantum import atoms, entropy, scales, spectra, utils, synthetics


###############################################################################
# Write a function that can generate a harmonic signal
# ----------------------------------------------------
# Let's write a function that can generate some simple blip-type sounds with
# decaying harmonics. It will take four arguments: T, the length of the output
# vector; num_intervals, the number of different blips; gamma, the exponential
# decay factor of the harmonic; random_state, a random seed to generate
# random pitches and phase shifts.
# The function proceeds by splitting the time length T into intervals, chooses
# base frequencies and phases, generates sinusoidal sounds and harmonics,
# and then adds a windowed version to the output signal.
def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = np.random.RandomState(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(1):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window

    return x


###############################################################################
# Let's take a look at what such a signal could look like
# -------------------------------------------------------
pow2_T = 13
T = 2 ** pow2_T
sample_rate = 80.
# x = generate_harmonic_signal(T)
x = synthetics.chirp_rdvxm_noise_16bit(duration_points=T, sample_rate_hz=sample_rate, noise_std_loss_bits=4.)
plt.figure(figsize=(8, 2))
plt.plot(x)
plt.title("Original signal")

###############################################################################
# Spectrogram
# -----------
# Let's take a look at the signal spectrogram
plt.figure(figsize=(8, 4))
plt.specgram(x, Fs=sample_rate)
plt.title("Time-Frequency spectrogram of signal")

###############################################################################
# Doing the scattering transform
# ------------------------------
# J = 6
# THE INVARIANCE SCALE IS THE SIGNAL DURATION!!
# SEE MATHWORKS: https://se.mathworks.com/help/wavelet/ug/wavelet-scattering.html
# PROVIDES A GOOD EXPLANATION, linear spacing on lower frequencies, log spacing on higher
J = pow2_T-8
Q = 24

scattering = Scattering1D(J, T, Q)
sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = \
    calibrate_scattering_filters(J, Q, r_psi=np.sqrt(0.5), sigma0=0.1, alpha=5.)
frequency_xi1 = np.array(xi1)*sample_rate
frequency_xi2 = np.array(xi2)*sample_rate

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

# Frequency in hz
freqs = np.array([tuple(xi * sample_rate) for xi in meta['xi']])

Sx = scattering(x)

# SEE: https://githubmemory.com/repo/kymatio/kymatio/issues/713
print("All Frequencies:", freqs)
print("Frequencies shape:", np.array(freqs).shape)
print("Sum of zeroth, first, and second order coefficients")
print(Sx[order0].shape[0]+Sx[order1].shape[0]+Sx[order2].shape[0])

print("First order Nth order frequencies:", frequency_xi1)
print(frequency_xi1.shape)
print(Sx[order1].shape)
print("Second order octave frequencies:", frequency_xi2)
print(frequency_xi2.shape)
print(Sx[order2].shape)

print("Break out frequency")
print("Frequencies shape:", np.array(freqs).shape)
print(order0)
print(order1)
print(order2)
# Get tuples
print(freqs[0, 0])
print(freqs[order1, 0])
print(freqs[order2, :])

#print(freqs[order2, 0]*freqs[order2, 1])

#exit()
plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(Sx[order0][0])
plt.title('Zeroth-order scattering')
plt.subplot(3, 1, 2)
plt.imshow(Sx[order1], aspect='auto')
plt.title('First-order scattering')
plt.subplot(3, 1, 3)
plt.imshow(Sx[order2], aspect='auto')
plt.title('Second-order scattering')

# Compare to spectrogram
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.pcolormesh(np.arange(Sx[order1].shape[1]), frequency_xi1, Sx[order1], shading='auto')
plt.ylim(np.min(frequency_xi1), np.max(frequency_xi1))
plt.title("First order")
plt.subplot(1, 2, 2)
plt.specgram(x, Fs=sample_rate)
plt.ylim(np.min(frequency_xi1), np.max(frequency_xi1))
plt.title('Spectrogram')

plt.show()
