"""
Plot the 1D wavelet filters
===========================
MAG: Map filters to frequencies and frequencies to sigma/scales
https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/filter_bank.py
"""

###############################################################################
# Preliminaries
# -------------
# Import the `scattering_filter_factory` method, which we will use
# to generate the filters.

from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.filter_bank import compute_xi_max, compute_params_filterbank, calibrate_scattering_filters

###############################################################################
# We then import `numpy` and `matplotlib` to display the filters.

import numpy as np
import math
import matplotlib.pyplot as plt


###############################################################################
# Filter parameters and generation
# --------------------------------
# The filters are defined for a certain support size `T` corresponds to
# the size of the input signal. `T` must be a power of two.
# TODO: DOCUMENT MINIMUM NO-WARNING VALUE OF T=2**10! e.g. T = 1024
T_log2 = 10
T = 2**T_log2

###############################################################################
# The parameter `J` specifies the maximum scale of the filters as a power of
# two. In other words, the largest filter will be concentrated in a time
# interval of size `2**J`. J <= T_log2
# TODO: DOCUMENT IMPORTANCE OF THIS PARAMETER - PROCESS FAILS IF THIS IS WRONG
# J = T_log2 - 1  # Would concentrates all the energy on the lower frequency; filters are horrible
# Geometric center of the passband, nice filters
J = T_log2//2


###############################################################################
# The `wavelet_order` parameter controls the number of wavelets per octave in the
# first-order filter bank.
# wavelet_order=1 is MINIMUM
wavelet_order = 6  # Named Q in Kymatio
frequency_ratio = 2**(1/wavelet_order)
quality_factor = 1/(2**(1/(2*wavelet_order)) - 2**(-1/(2*wavelet_order)))
###############################################################################
# Note that it is currently not possible to control the number of wavelets
# per octave in the second-order filter bank, which is fixed to one.

# The highest frequency is returned by xi_max
# xi_max = max(1. / (1. + math.pow(2., 3. / wavelet_order)), 0.35)
# TODO: Scale to produce standard bands
xi_max = compute_xi_max(Q=wavelet_order)  # Approaches 0.5 (Nyquist) as wavelet_order increases
print("Order:", wavelet_order)
print("Theoretical quality factor:", quality_factor)
print("Theoretical fractional bandwidth:", 1/quality_factor)
print('Max frequency xi_max:', xi_max)

# From code, function compute_sigma_psi:
#     r : float, optional
#         Positive parameter defining the bandwidth to use.
#         Should be < 1. We recommend keeping the default value.
#         The larger r, the larger the filters in frequency domain.
r = math.sqrt(0.5)
factor = 1. / math.pow(2, 1. / wavelet_order)
term1 = (1 - factor) / (1 + factor)
term2 = 1. / math.sqrt(2 * math.log(1. / r))
sigma_to_xi = term1 * term2

# Inspect source code: in filter_bank.py
sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = \
    calibrate_scattering_filters(J=J, Q=wavelet_order, r_psi=np.sqrt(0.5), sigma0=0.1, alpha=5.)


# sigma is the argument of the Gaussian
print("\nFirst order scaled frequencies, xi1: (Nth Octave)")
print(xi1)
print("Theoretical xi1 ratio:", frequency_ratio)
print("xi1 ratio:", np.array(xi1[:-1])/np.array(xi1[1:]))
print("coded ratio of sigma_to_xi:", sigma_to_xi)
print("sigma1/xi1:", np.array(sigma1)/np.array(xi1))
print("\nSecond order scaled frequencies, xi2 (Octave):")
print(xi2)
print("sigma2/xi2:", np.array(sigma2)/np.array(xi2))

# Evaluate morlet filters with morlet_1d
# TODO: reconcile with morlet_1d
#  gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
# TODO: T vs log2(T) - document inconsistent syntax (2**J)
phi_f, psi1_f, psi2_f, t_support = scattering_filter_factory(np.log2(T), J, Q=wavelet_order)

###############################################################################
# The `phi_f` output is a dictionary where each integer key corresponds points
# to the instantiation of the filter at a certain resolution. In other words,
# `phi_f[0]` corresponds to the lowpass filter at resolution `T`, while
# `phi_f[1]` corresponds to the filter at resolution `T/2`, and so on.
#
# While `phi_f` only contains a single filter (at different resolutions),
# the `psi1_f` and `psi2_f` outputs are lists of filters, one for each wavelet
# bandpass filter in the filter bank.

###############################################################################
# Plot the filters
# ================
# We are now ready to plot the filters. We first display the lowpass filter
# (at full resolution) in red. We then plot each of the bandpass filters in
# blue. Since we don't care about the negative frequencies, we limit the
# plot to the frequency interval :math:`[0, 0.5]`. Finally, we add some
# explanatory labels and title.
# The frequency map in the script seems wrong; the number of filters is correct
# OMEGA = F/FS, FS = Sample rate
omega = np.arange(T)/T  # Goes from zero to unity, where unity is the sample rate
omega_log2 = np.log2(omega + 2**-16)  # Nyquist is at log2(2**-1) = -1

plt.figure()
plt.plot(omega_log2, phi_f[0], 'r')
for psi_f in psi1_f:
    plt.plot(omega_log2, psi_f[0], 'b')
plt.grid(True)
plt.xlim(omega_log2[1], np.log2(0.5))

plt.xlabel(r'log2(f/fs)', fontsize=18)
plt.ylabel(r'$\hat\psi_j(f )$', fontsize=18)
plt.title('First-order filters (wavelet_order = {})'.format(wavelet_order), fontsize=18)

###############################################################################
# Do the same plot for the second-order filters. Note that since here `wavelet_order = 1`,
# we obtain wavelets that have higher frequency bandwidth.

plt.figure()
plt.plot(omega_log2, phi_f[0], 'r')
for psi_f in psi2_f:
    plt.plot(omega_log2, psi_f[0], 'b')
plt.xlim(omega_log2[1], np.log2(0.5))
# plt.xlim(omega_log2[1], omega_log2[-1])
plt.grid(True)
plt.xlabel(r'log2(f/fs)', fontsize=18)
plt.ylabel(r'$\hat\psi_j(f )$', fontsize=18)
plt.title('Second-order filters (wavelet_order = 1)', fontsize=18)
###############################################################################
# Display the plots!

plt.show()
