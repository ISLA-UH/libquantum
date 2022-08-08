"""
STRIPPED DOWN STOCKWELL CODE
"""

import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


# CONSTANT
MN_over_N = 3 * np.pi / 4


def log2(x):
    """
    Returns log2 of x
    :param x: input
    :return: log2(x), same as np.log2
    """
    # log_b(x) = log_d(x)/log_d(b); log_2(x) = log_e(x)/log_e(2)
    return np.log(x) / np.log(2)


def stx_octave_band_frequencies(
    band_order_Nth: float, frequency_sample_rate_hz: float, frequency_averaging_hz: float
) -> np.ndarray:
    """
    Compute standardized octave band frequencies
    :param band_order_Nth:
    :param frequency_sample_rate_hz:
    :param frequency_averaging_hz:
    :return: frequency_octaves_hz
    """

    cycles_M = MN_over_N * band_order_Nth
    # Find highest power of two to match averaging frequency
    averaging_window: float = cycles_M / frequency_averaging_hz
    averaging_points: float = averaging_window * frequency_sample_rate_hz
    # Find J for 2^J points of averaging window
    J: int = int(np.ceil(log2(averaging_points)))
    # Find K for 2^K points of Order
    K: int = int(np.ceil(log2(band_order_Nth / 3) + 3))
    band_n_start = int(band_order_Nth)
    band_n_stop = int((J - K) * band_order_Nth)
    band_n = np.arange(band_n_start, band_n_stop + 1)
    # Take it up to Nyquist
    frequency_octaves_hz_Nyq = 2 ** (-band_n / band_order_Nth) * frequency_sample_rate_hz
    # Remove the Nyquist band at zeroth index, and then flip order for ascending frequencies
    frequency_octaves_hz = np.flip(frequency_octaves_hz_Nyq[1:])
    return frequency_octaves_hz


def stx_linear_frequencies(
    band_order_Nth: float, frequency_sample_rate_hz: float, frequency_averaging_hz: float
) -> np.ndarray:
    """
    Compute same number of linear bands as in octave bands over the same bandpass
    :param band_order_Nth:
    :param frequency_sample_rate_hz:
    :param frequency_averaging_hz:
    :return: frequency_octaves_hz
    """
    frequency_octaves_hz = stx_octave_band_frequencies(band_order_Nth, frequency_sample_rate_hz, frequency_averaging_hz)
    frequency_lin_min = frequency_octaves_hz[0]  # First band
    frequency_lin_max = frequency_octaves_hz[-1]  # Last band
    frequency_linear_hz = np.linspace(start=frequency_lin_min, stop=frequency_lin_max, num=len(frequency_octaves_hz))
    return frequency_linear_hz


def stx_power_any_scale_pow2(
    band_order_Nth: float, sig_wf: np.ndarray, frequency_sample_rate_hz: float, frequency_stx_hz: np.ndarray
) -> np.ndarray:
    """
    With assumptions and simplifications
    :param band_order_Nth: Fractional octave band - revisit
    :param sig_wf: input signal with n_fft_pow2 = 2^M points
    :param frequency_sample_rate_hz: sample rate in Hz
    :param frequency_stx_hz: frequency vector in increasing order, NSTX number of frequencies
    :return: frequency_stx_hz, time_stx_s, tfr_stx_power
    """

    n_stx_freq: int = len(frequency_stx_hz)  # Number of stx frequencies
    # The numpy np.arange method creates a vector of integers, starting at zero
    cycles_M = MN_over_N * band_order_Nth
    # Compute the nondimensionalized stx frequencies and scales
    frequency_stx = frequency_stx_hz / frequency_sample_rate_hz
    omega_stx = 2 * np.pi * frequency_stx  # non-dimensional angular stx frequency
    sigma_stx = cycles_M / omega_stx

    n_fft_pow2: int = len(sig_wf)  # Number of fft points
    # Compute FFT and concatenate
    sig_fft = fft(sig_wf)
    # Frequency order depends on the method.
    # In scipy, the frequency cycles goes to Nyquist, flips sign to negative,
    # and then ascends (less negative) to zero. In scipy:
    # frequency_fft = fftfreq(n_fft_pow2, 1/frequency_sample_rate_hz)   # in units of 1/sample interval
    # Since we are not returning the FFT frequencies, we keep their nondimensionalized form
    # The positive fft frequencies can be computed from:
    frequency_fft_pos = np.arange(n_fft_pow2 // 2 + 1) / len(sig_wf)  # Up to Nyquist
    # The negative fft frequencies can be computed from:
    frequency_fft_neg = -np.flip(frequency_fft_pos[1:-1])  # Ascends to zero
    frequency_fft = np.concatenate([frequency_fft_pos, frequency_fft_neg], axis=-1)
    omega_fft = 2 * np.pi * frequency_fft  # scaled angular frequency

    # This next step is not obvious unless one thinks of the folding frequency.
    # The frequency on the concatenated array cycles from 0 to Nyquist, then
    # to negative frequencies, then again from 0 to Nyquist, and back to negative
    sig_fft_cat = np.concatenate([sig_fft, sig_fft], axis=-1)

    # Initialize: In python, the real is float64 and the imaginary is float64, yielding complex128
    # Don't know how this work in other languages
    tfr_stx = np.empty((n_stx_freq, n_fft_pow2), dtype=np.complex128)

    # Minimal iteration over the stx bands
    for isx in range(n_stx_freq):
        fsx = frequency_stx[isx]
        # This is real; note the envelope follows omega, and peaks at index [0].
        # Thus the shifted FFT (sig_fft_shifted) peaks at the first point
        gaussian_envelope = np.exp(-0.5 * (sigma_stx[isx] ** 2.0) * (omega_fft**2.0))
        # *** This is the main event: find the closest fft frequency to fsx. Not sure how this works in C/Java
        stx_index = np.abs(frequency_fft - fsx).argmin()  # Locks on to positive frequency
        # The next step takes the nfft_pow_2 frequency components that follow the match on the positive frequency
        # It is mathematically equivalent to X(f_fft + f_stx). Different fft implementations may behave differently
        sig_fft_shifted = sig_fft_cat[stx_index : stx_index + n_fft_pow2]
        # Multiply by the (real) Gaussian (peaking at the first pont),
        # take the ifft, and we have the Stockwell transform
        tfr_stx[isx, :] = ifft(sig_fft_shifted * gaussian_envelope)

    # The scaling below returns the correct value for a sine wave test
    tfr_stx_power = 2 * np.abs(tfr_stx) ** 2
    # The stx time vector has the same granularity and dimensionality as the input signal vector
    # The stx frequency vector is also unchanged.

    return tfr_stx_power


if __name__ == "__main__":
    # Construct sine wave with 2^16 points (>60 hz at 800 hz) and center frequency frequency_sig_hz
    print("STX function check")
    n_fft: int = 2**16  # Signal duration in points
    order = 12
    frequency_fs_hz = 800.0
    frequency_sig_hz = 50.0
    frequency_min_hz = 25.0
    time_s = np.arange(n_fft) / frequency_fs_hz
    sig_cw = np.sin(2 * np.pi * frequency_sig_hz * time_s)

    # Fractional octave band frequencies
    frequency_nth_hz = stx_octave_band_frequencies(
        band_order_Nth=order, frequency_sample_rate_hz=frequency_fs_hz, frequency_averaging_hz=frequency_min_hz
    )

    # The lin frequency algo misses the signal frequency, but will look OK normalized.
    # Advantage: same number of frequencies as above = same grid.
    # Can refine to lock onto signal frequency if known.
    frequency_lin_hz = stx_linear_frequencies(
        band_order_Nth=order, frequency_sample_rate_hz=frequency_fs_hz, frequency_averaging_hz=frequency_min_hz
    )

    # # UNCOMMENT TO TEST: Designed to hit the signal frequency, returns correct power
    # frequency_lin_hz = np.arange(25, frequency_fs_hz/2, 5)

    print("Fractional Octave Band Order:", order)
    # print("STX Nth Octave Frequencies:", frequency_nth_hz)
    print("Number of Nth Octave Frequencies:", len(frequency_nth_hz))
    # print("STX Linear Frequencies:", frequency_lin_hz)
    print("Number of Linear Frequencies:", len(frequency_lin_hz))

    tfr_power_nth = stx_power_any_scale_pow2(
        band_order_Nth=order, sig_wf=sig_cw, frequency_sample_rate_hz=frequency_fs_hz, frequency_stx_hz=frequency_nth_hz
    )

    tfr_power_lin = stx_power_any_scale_pow2(
        band_order_Nth=order, sig_wf=sig_cw, frequency_sample_rate_hz=frequency_fs_hz, frequency_stx_hz=frequency_lin_hz
    )

    plt.figure()
    plt.pcolormesh(time_s, frequency_nth_hz, tfr_power_nth)
    plt.yscale("log")
    plt.ylabel("Log Frequency, hz")
    plt.xlabel("Time, s")
    plt.colorbar()
    plt.title("LOG: Theoretical Sine Wave Power = 1/2")

    plt.figure()
    plt.pcolormesh(time_s, frequency_lin_hz, tfr_power_lin)
    plt.yscale("linear")
    plt.ylabel("Lin Frequency, hz")
    plt.xlabel("Time, s")
    plt.colorbar()
    plt.title("LIN: Theoretical Sine Wave Power = 1/2")

    plt.show()
