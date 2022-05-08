import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq, fftshift
from libquantum.scales import EPSILON
# TODO: Construct Styx repo


def sig_pad_up_to_pow2(sig_wf, n_fft):
    """Aux function."""
    # flatten to 2 D and memorize original shape
    n_times = sig_wf.shape[-1]

    # This is a clever bit of code
    def _is_power_of_two(n):
        return not (n > 0 and (n & (n - 1)))

    if n_fft is None or (not _is_power_of_two(n_fft) and n_times > n_fft):
        # Compute next power of 2
        n_fft = 2 ** int(np.ceil(np.log2(n_times)))
    elif n_fft < n_times:
        raise ValueError("n_fft cannot be smaller than signal size. "
                         "Got %s < %s." % (n_fft, n_times))
    if n_times < n_fft:
        print('The input signal is shorter ({}) than "n_fft" ({}). '
              'Applying zero padding.'.format(sig_wf.shape[-1], n_fft))
        zero_pad: int = n_fft - n_times
        pad_array = np.zeros(sig_wf.shape[:-1] + (zero_pad,), sig_wf.dtype)
        sig_wf = np.concatenate((sig_wf, pad_array), axis=-1)
    else:
        zero_pad: int = 0

    return sig_wf, n_fft, zero_pad


def tfr_stockwell(sig_wf: np.ndarray,
                  time_sample_interval: float,
                  nth_order: float = 8.0,
                  n_fft_in: int = None,
                  frequency_min: float = None,
                  frequency_max: float = None,
                  frequency_step: float = None,
                  frequency_is_geometric: bool = False,
                  frequency_in_inferno: bool = False):

    frequency_sample_rate = 1 / time_sample_interval
    cycles_per_frequency = 12 / 5 * nth_order

    # Compute the nearest higher power of two number of points for the fft
    sig_wf_pow2, n_fft_pow2, zero_pad = sig_pad_up_to_pow2(sig_wf, n_fft_in)

    # For reduction back to the near original signal after taking the ifft. Should match input sig.
    n_fft_out = n_fft_pow2 - zero_pad

    # Transformations are on zero padded signals from here onwards
    # Take FFT and concatenate. A leaner version would let the fft do the padding.
    X = fft(sig_wf_pow2)
    XX = np.concatenate([X, X], axis=-1)

    frequency_fft = fftfreq(n_fft_pow2, time_sample_interval)   # in units of 1/sample interval
    mu_fft = 2 * np.pi * frequency_fft / frequency_sample_rate  # scaled angular frequency

    window_longest_time = n_fft_pow2 / frequency_sample_rate
    frequency_min_nth = cycles_per_frequency/window_longest_time

    # Initialize stx frequencies
    if frequency_min is None:
        frequency_min = frequency_min_nth
    if frequency_max is None:
        frequency_max = frequency_sample_rate / 2.

    # TODO: Standardize
    # Computing nearest frequency later on anyway, and then using that to compute the fft.
    start_f_idx = np.abs(frequency_fft - frequency_min).argmin()
    stop_f_idx = np.abs(frequency_fft - frequency_max).argmin()
    f_start = frequency_fft[start_f_idx]
    f_stop = frequency_fft[stop_f_idx]

    if frequency_step is None:
        # Only for linear - may want to push this code downstream
        # Reduce the fft resolution by a factor of two
        frequency_step = 2 * (frequency_max - frequency_min) / len(frequency_fft)

    if frequency_is_geometric is True:
        num_octaves = np.log2(f_stop/f_start)
        num_bands = int(num_octaves * nth_order)
        print("Number of bands:", num_bands)
        frequency_stx = np.logspace(np.log2(f_start), np.log2(f_stop), num=num_bands, base=2.)
    else:
        frequency_stx = np.arange(f_start, f_stop, frequency_step)
    print("Shape of frequency_stx", frequency_stx.shape)
    print("SX Band edges:", frequency_stx[0], frequency_stx[-1])

    # Construct shifting frequency indexes
    frequency_stx_fft = np.empty(len(frequency_stx))

    # Construct time domain and fft of window
    windows_fft = np.empty((len(frequency_stx), n_fft_pow2), dtype=np.complex128)
    # tfr_stx_pow2 = np.empty(1, n_fft_pow2, dtype=np.complex128)

    tfr_stx = np.empty((len(frequency_stx), n_fft_out), dtype=np.complex128)
    psd_stx = np.empty((len(frequency_stx), n_fft_out))

    for isx, fsx in enumerate(frequency_stx):
        # TODO: Precompute as many variables as possible before for loop
        stx_index = np.abs(frequency_fft - fsx).argmin()
        frequency_stx_fft[isx] = frequency_fft[stx_index]
        # TODO: Verify
        # nu_sx = 2*np.pi*fsx/sample_rate    # non-dimensional angular stx frequency
        nu_sx = 2 * np.pi * frequency_stx_fft[isx] / frequency_sample_rate    # eq non-dimensional angular fft frequency
        if nu_sx == 0.:
            windows_fft[isx] = np.ones(len(n_fft_pow2))
        else:
            sigma = cycles_per_frequency/nu_sx
            windows_fft[isx] = np.exp(-0.5 * (sigma ** 2.) * (mu_fft ** 2.))

        # This is it
        tfr_stx_pow2 = ifft(XX[stx_index:stx_index + n_fft_pow2] * windows_fft[isx])
        if zero_pad > 0:
            tfr_stx[isx, :] = tfr_stx_pow2[:-zero_pad:1]
        else:
            tfr_stx[isx, :] = tfr_stx_pow2
        # Power
        tfr_abs = np.abs(tfr_stx[isx, :])**2
        psd_stx[isx, :] = tfr_abs + EPSILON

    return tfr_stx, psd_stx, frequency_stx, frequency_stx_fft, windows_fft
