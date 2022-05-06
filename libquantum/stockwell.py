import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq, fftshift
from libquantum.scales import EPSILON


def tfr_array_stockwell_isla(data, sample_rate, fmin=None, fmax=None, n_fft=None, df=None,
                             order=8.0, binary_order: bool = False):

    M = 12/5*order
    sample_interval = 1/sample_rate
    if n_fft is None:
        n_fft = data.shape[-1]

    Tw = n_fft/sample_rate
    fmin_nth = 12/5 * order/Tw

    if fmin is None:
        fmin = fmin_nth
    if fmax is None:
        fmax = sample_rate/2.
    if df is None:
        df = (fmax-fmin)/12.

    # Take FFT and concatenate
    X = fft(data)
    XX = np.concatenate([X, X], axis=-1)
    freqs_fft = fftfreq(n_fft, sample_interval)
    mu_fft = 2*np.pi*freqs_fft/sample_rate

    # TODO: Standardize
    start_f_idx = np.abs(freqs_fft - fmin).argmin()
    stop_f_idx = np.abs(freqs_fft - fmax).argmin()
    f_start = freqs_fft[start_f_idx]
    f_stop = freqs_fft[stop_f_idx]
    if binary_order is True:
        num_octaves = np.log2(f_stop/f_start)
        num_bands = int(num_octaves*order)
        print("Number of bands:", num_bands)
        freqs_s = np.logspace(np.log2(f_start), np.log2(f_stop), num=num_bands, base=2.)
    else:
        freqs_s = np.arange(f_start, f_stop, df)
    print("Shape of freqs_s", freqs_s.shape)
    print("SX Band edges:", freqs_s[0], freqs_s[-1])

    # TODO: Consolidate all the loops over freqs_s
    # Construct shifting frequency indexes
    freqs_s_index = np.empty(len(freqs_s), dtype=int)
    for isx, fsx in enumerate(freqs_s):
        freqs_s_index[isx] = np.abs(freqs_fft - freqs_s[isx]).argmin()

    # TODO: find fft frequencies that exactly match - necessary for shifting FFT
    # Gabor window
    # TODO: Reconcile with above granularity
    nu_s = 2*np.pi*freqs_s/sample_rate  # Nondimensionalized angular frequency
    windows_f3 = np.empty((len(freqs_s), len(data)), dtype=np.complex128)
    # TODO: These can be constructed as a matrix w/o for loop
    for i_f, nu in enumerate(nu_s):
        if nu == 0.:
            windows_f3[i_f] = np.ones(len(data))
        else:
            sigma = M/nu
            windows_f3[i_f] = np.exp(-0.5 * (sigma ** 2.)*(mu_fft ** 2.))

    psd3 = np.empty((len(windows_f3), n_fft))
    # Frequency shift the FFT, then compute the IFFT
    for i_f, window in enumerate(windows_f3):
        # this is f_fft + f_s
        f_idx = freqs_s_index[i_f]
        TFR = ifft(XX[f_idx:f_idx + n_fft] * window) + EPSILON
        TFR_abs = np.abs(TFR)
        TFR_abs *= TFR_abs  # Square
        psd3[i_f, :] = TFR_abs
    print("W3 shape:", windows_f3.shape)
    return psd3, freqs_s, freqs_fft, windows_f3
