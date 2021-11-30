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
import kymatio_22.kymatio_atom_utils as atom

###############################################################################
# Modified with a more careful handling of the initialization noise - not completely random.
# The noise model is critical to the reconstruction process

Q = 6
sample_rate = 8000.

# Min spec of 2**10, critically sampled
log2T = 10
T = 2**log2T

###############################################################################
# Prepare the scattering transform
# ------------------------------

# GT blast input signal
x_sig, gt_J, gt_frequency_hz = \
    atom.gt_blast_kymatio(time_points=T,
                          nth_octave=Q,
                          sig_octaves_up=2,
                          noise_bit_loss=16,
                          sample_rate_hz=sample_rate)

J = gt_J



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
sig_gauss_std = x_sig_std/8
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

# Q = 24  # Nth=Q order, default of 16
# J = 6   # Primary scale = 2**J

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
