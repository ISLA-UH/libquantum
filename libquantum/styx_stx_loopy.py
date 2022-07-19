import numpy as np
from scipy.fft import fft, ifft, fftfreq
from libquantum import styx_cwt
from typing import Tuple


def log2(x):
    """
    Returns log2 of x
    :param x: input
    :return: log2(x), same as np.log2
    """
    # log_b(x) = log_d(x)/log_d(b); log_2(x) = log_e(x)/log_e(2)
    return np.log(x)/np.log(2)


def stx_octave_band_frequencies(band_order_Nth: float,
                                frequency_sample_rate_hz: float,
                                frequency_averaging_hz: float) -> np.ndarray:
    """
    Compute standardized octave band frequencies
    :param band_order_Nth:
    :param frequency_sample_rate_hz:
    :param frequency_averaging_hz:
    :return: frequency_octaves_hz
    """
    # Find highest power of two to match averaging frequency
    averaging_period = 1/frequency_averaging_hz
    averaging_points = averaging_period*frequency_sample_rate_hz
    averaging_points_pow2 = 2**int(np.ceil(log2(averaging_points)))
    print(averaging_points_pow2)


def stx_complex_any_scale_pow2(band_order_Nth: float,
                               sig_wf: np.ndarray,
                               frequency_sample_rate_hz: float,
                               frequency_stx_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    With assumptions and simplifications
    :param band_order_Nth: Fractional octave band - revisit
    :param sig_wf: input signal with n_fft_pow2 = 2^M points
    :param frequency_sample_rate_hz: sample rate in Hz
    :param frequency_stx_hz: frequency vector in increasing order, NSTX number of frequencies
    :return: frequency_stx_hz, time_stx_s, tfr_stx_power
    """
    n_fft_pow2: int = len(sig_wf)  # Number of fft points
    n_stx_freq: int = len(frequency_stx_hz)  # Number of stx frequencies
    time_stx_s = np.arange(n_fft_pow2)/frequency_sample_rate_hz
    cycles_M = 3*np.pi/4 * band_order_Nth

    # Compute FFT and concatenate
    sig_fft = fft(sig_wf)
    sig_fft_cat = np.concatenate([sig_fft, sig_fft], axis=-1)
    # Frequency order depends on the method
    frequency_fft = fftfreq(n_fft_pow2, 1/frequency_sample_rate_hz)   # in units of 1/sample interval
    print(frequency_fft)

    omega_fft = 2 * np.pi * frequency_fft / frequency_sample_rate_hz  # scaled angular frequency
    omega_stx = 2*np.pi*frequency_stx_hz/frequency_sample_rate_hz    # non-dimensional angular stx frequency
    sigma_stx = cycles_M/omega_stx

    # Construct 2d matrices
    # sig_fft_cat_2d = np.tile(sig_fft_cat, (scale_points, 1))
    omega_fft_2d = np.tile(omega_fft, (n_stx_freq, 1))
    sigma_stx_2d = np.tile(sigma_stx, (n_fft_pow2, 1)).T
    windows_fft_2d = np.exp(-0.5 * (sigma_stx_2d ** 2.) * (omega_fft_2d ** 2.))

    # Construct shifting frequency indexes
    tfr_stx = np.empty((n_stx_freq, n_fft_pow2), dtype=np.complex128)

    # Minimal iteration
    for isx, fsx in enumerate(frequency_stx_hz):
        # This is the main event
        stx_index = np.abs(frequency_fft - fsx).argmin()
        tfr_stx[isx, :] = ifft(sig_fft_cat[stx_index:stx_index + n_fft_pow2] * windows_fft_2d[isx, :])

    return frequency_stx_hz, time_stx_s, tfr_stx


if __name__ == "__main__":
    print('STX function check')