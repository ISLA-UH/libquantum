from kymatio.numpy import Scattering1D
import matplotlib.pyplot as plt
import numpy as np

from libquantum import synthetics as synth
from libquantum import blast_pulse as kaboom
from kymatio.scattering1d.filter_bank import calibrate_scattering_filters

# From plot_synthetic
# https://www.kymat.io/userguide.html#id3
# https://www.kymat.io/gallery_1d/plot_synthetic.html#sphx-glr-gallery-1d-plot-synthetic-py


def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap. From Kymatio example.
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


def gt_blast_kymatio(time_points: int = 2**13,
                     nth_octave: int = 6,
                     sig_octaves_up: float = 2,
                     noise_bit_loss: float = 6,
                     sample_rate_hz: float = 1024):

    number_oscillations = 2.355*nth_octave
    max_period_points = time_points/number_oscillations
    log2_period_points = np.floor(np.log2(max_period_points)) - sig_octaves_up

    pseudo_period_s = (2**log2_period_points)/sample_rate_hz
    sig_frequency_hz = 1/pseudo_period_s

    time_s = np.arange(time_points)/sample_rate_hz
    time_half_s = np.max(time_s)/2.
    time_shifted_s = time_s - time_half_s

    # Build signal, no noise
    sig_theory = kaboom.gt_blast_period_center(time_center_s=time_shifted_s,
                                               pseudo_period_s=pseudo_period_s)
    # Add white noise
    # Variance computed from transient, stressing at bit_loss=1

    sig_noise = synth.white_noise_fbits(sig=sig_theory,
                                        std_bit_loss=noise_bit_loss)
    gt_white = sig_theory + sig_noise

    # AA filter of signal with noise
    sig = synth.antialias_halfNyquist(synth=gt_white)  # With noise
    sig_0 = sig.copy().astype('float32')

    return sig_0, int(log2_period_points), sig_frequency_hz
