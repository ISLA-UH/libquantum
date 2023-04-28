"""
Reconstruct a synthetic signal from its scattering transform
============================================================
In this example we generate a harmonic signal of a few different frequencies,
analyze it with the 1D scattering transform, and reconstruct the scattering
transform back to the harmonic signal.
"""

###############################################################################
# Import the necessary packages
# -----------------------------

import numpy as np
import torch
import scipy.signal as signal
from kymatio.torch import Scattering1D

from torch.autograd import backward
import matplotlib.pyplot as plt

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
# Modified with a more careful handling of the initialization noise - not completely random.
# The noise model is critical to the reconstruction process

# Key specs
sample_rate = 1024
log2T = 13
# Signal duration in number of points
T = 2 ** 13

# This is the input signal
x_sig = generate_harmonic_signal(T)
# TODO: Play with taper
win_taper = np.hanning(T).astype('float32')

# Initialization noise model (and levels) are important
# Random guess to initialize. This does not work well.
# torch.manual_seed(0)
# y = torch.randn((T,), requires_grad=True)
# Modify the noise specification

# TODO: Play with model noise and signal noise std
# Gaussian noise model
# Add to signal
x_sig_std = np.std(x_sig)
sig_gauss_std = x_sig_std/32
sig_gauss_noise = np.random.normal(0, sig_gauss_std, size=T).astype('float32')
x0 = (x_sig + sig_gauss_noise)*win_taper

# Assign standard deviation
# y_std = 10*sig_std
# Noise from data std

y_std = sig_gauss_std
y_noise = np.random.normal(0, y_std, size=T).astype('float32')
y0 = y_noise*win_taper

# x = torch.from_numpy(x0)  # Original
x = torch.tensor(x0, requires_grad=False)  # Match below
y = torch.tensor(y0, requires_grad=True)

# Plot TDR
plt.figure(figsize=(8, 2))
plt.plot(x.numpy())
plt.title("Original signal")

plt.figure(figsize=(8, 2))
plt.plot(y.detach().numpy())
plt.title("Noise")

###############################################################################
# Signal spectrogram.

plt.figure(figsize=(8, 8))
plt.specgram(x.numpy(), Fs=sample_rate)
plt.title("Spectrogram of original signal")
plt.colorbar()

# Noise spectrogram
plt.figure(figsize=(8, 8))
plt.specgram(y.detach().numpy(), Fs=sample_rate)
plt.title("Spectrogram of initialization noise")
plt.colorbar()

###############################################################################
## Scattering transform

Q = 24  # Nth=Q order, default of 16
J = 6   # Primary scale = 2**J


scattering = Scattering1D(J, T, Q)

# MUST use same scattering parameters
Sx = scattering(x)
Sy = scattering(y)

# TODO: Explore
learning_rate = 100
bold_driver_accelerator = 1.1
bold_driver_brake = 0.55
n_iterations = 100  # 200 in example

###############################################################################
# Reconstruct the scattering transform back to original signal.

history = []
signal_update = torch.zeros_like(x)

# Iterate to reconstruct random guess to be close to target.
for k in range(n_iterations):
    # Backpropagation.
    err = torch.norm(Sx - Sy)

    if k % 10 == 0:
        print('Iteration %3d, loss %.2f' % (k, err.detach().numpy()))

    # Measure the new loss.
    # Does not run: history.append(err)
    history.append(err.detach().numpy())

    backward(err)

    delta_y = y.grad

    # Gradient descent
    with torch.no_grad():
        signal_update = - learning_rate * delta_y
        new_y = y + signal_update
    new_y.requires_grad = True

    # New forward propagation.
    Sy = scattering(new_y)

    if history[k] > history[k - 1]:
        learning_rate *= bold_driver_brake
    else:
        learning_rate *= bold_driver_accelerator
        y = new_y

plt.figure(figsize=(8, 2))
plt.semilogy(history)
plt.title("MSE error vs. iterations")

plt.figure(figsize=(8, 2))
plt.plot(y.detach().numpy())
plt.title("Reconstructed signal")

plt.figure(figsize=(8, 8))
plt.specgram(y.detach().numpy(), Fs=sample_rate)
plt.title("Spectrogram of reconstructed signal")
plt.colorbar()

plt.show()
