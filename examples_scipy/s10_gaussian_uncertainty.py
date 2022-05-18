"""
libquantum example: gaussian uncertainty
Evaluate time-frequency marginals for Gaussian envelopes
Introduce the concept of a Q-driven STFT

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import atoms, entropy, scales, synthetics, utils  # spectra,
# import libquantum.plot_templates.plot_time_frequency_reps as pltq  # white background
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
from typing import Tuple

print(__doc__)

if __name__ == "__main__":
    frequency_sample_rate_hz = 100.
    frequency_center_hz = 10.
    order_nth = 12.

    # Prepare to iterate
    q = 1
    p = 2
    r = 1

    # Stage is set
    number_cycles = 3*np.pi/4*order_nth

    # Scaled frequency-time
    frequency_center = frequency_center_hz/frequency_sample_rate_hz
    period_center = 1./frequency_center
    time_duration_nd = number_cycles*period_center
    time_nd = np.arange(int(time_duration_nd))
    time_nd = time_nd - time_nd[-1]/2

    scale_atom = (3*order_nth/8)*period_center
    scale_beta = (1 + q * frequency_center**p)*frequency_center**(1-r)

    scale_stx = scale_atom*scale_beta

    gauss_envelope = np.exp(-0.5*(time_nd/scale_stx)**2)

    plt.plot(time_nd/time_duration_nd, gauss_envelope)
    plt.grid(True)
    plt.show()
